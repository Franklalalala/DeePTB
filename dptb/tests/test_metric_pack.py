"""STAGE 0 wire-compatibility tests for the typed metric packs.

These pin the named-field packs (``MetricPack`` / ``DynamicBatchStat`` /
``ExpertDisplayMetric``) to the exact legacy magic-int layout that
``MultiTrainer`` ships through ``all_reduce`` / ``all_gather``. If any of these
fail, the refactor changed the bytes on the wire and distributed training would
silently diverge (or deadlock).
"""

import pytest
import torch

from dptb.nnops.metric_pack import (
    MetricPack,
    DynamicBatchStat,
    ExpertDisplayMetric,
)
from dptb.nnops.multi_trainer import MultiTrainer


# ---------------------------------------------------------------------------
# Canonical (hard-coded) slot maps — the SPEC, independent of any source const.
# ---------------------------------------------------------------------------

_METRIC_SLOTS = {
    "loss_opt_sum": 0,
    "onsite_weighted_sum": 1,
    "hopping_weighted_sum": 2,
    "active_nodes_sum": 3,
    "active_edges_sum": 4,
    "onsite_l1_sum": 5,
    "onsite_mse_sum": 6,
    "onsite_cnt_sum": 7,
    "hopping_l1_sum": 8,
    "hopping_mse_sum": 9,
    "hopping_cnt_sum": 10,
    "z_sum": 11,
    "z_cnt": 12,
    "cv_sum": 13,
    "cv_cnt": 14,
    "grad_norm_sum": 15,
    "step_count": 16,
}

_DB_SLOTS = {
    "num_graphs_sum": 0,
    "cost_sum": 1,
    "num_nodes_sum": 2,
    "num_edges_sum": 3,
    "max_item_cost_sum": 4,
    "step_count": 5,
    "oom_skipped_count": 6,
}

_DISPLAY_SLOTS = {
    "expert_onsite": 0,
    "expert_hopping": 1,
    "grad_norm": 2,
    "lr": 3,
    "active_nodes": 4,
    "active_edges": 5,
}


# ---------------------------------------------------------------------------
# Index pinning against the hard-coded spec.
# ---------------------------------------------------------------------------

def test_metric_pack_indices_match_spec():
    assert MetricPack.LENGTH == 17
    assert MetricPack.length() == 17
    for name, slot in _METRIC_SLOTS.items():
        assert MetricPack.index(name) == slot, name


def test_dynamic_batch_stat_indices_match_spec():
    assert DynamicBatchStat.LENGTH == 7
    assert DynamicBatchStat.length() == 7
    for name, slot in _DB_SLOTS.items():
        assert DynamicBatchStat.index(name) == slot, name


def test_expert_display_metric_indices_match_spec():
    assert ExpertDisplayMetric.LENGTH == 6
    assert ExpertDisplayMetric.length() == 6
    for name, slot in _DISPLAY_SLOTS.items():
        assert ExpertDisplayMetric.index(name) == slot, name


# ---------------------------------------------------------------------------
# Index pinning against the RETAINED legacy MultiTrainer constants. This proves
# the typed pack agrees with the exact integers the shipping code used.
# ---------------------------------------------------------------------------

def test_metric_pack_indices_match_legacy_multitrainer_constants():
    assert MetricPack.LENGTH == MultiTrainer._PACK_LEN
    assert MetricPack.index("loss_opt_sum") == MultiTrainer._P_LOSS_OPT_SUM
    assert MetricPack.index("onsite_weighted_sum") == MultiTrainer._P_ONSITE_WEIGHTED_SUM
    assert MetricPack.index("hopping_weighted_sum") == MultiTrainer._P_HOPPING_WEIGHTED_SUM
    assert MetricPack.index("active_nodes_sum") == MultiTrainer._P_ACTIVE_NODES_SUM
    assert MetricPack.index("active_edges_sum") == MultiTrainer._P_ACTIVE_EDGES_SUM
    assert MetricPack.index("onsite_l1_sum") == MultiTrainer._P_ONSITE_L1_SUM
    assert MetricPack.index("onsite_mse_sum") == MultiTrainer._P_ONSITE_MSE_SUM
    assert MetricPack.index("onsite_cnt_sum") == MultiTrainer._P_ONSITE_CNT_SUM
    assert MetricPack.index("hopping_l1_sum") == MultiTrainer._P_HOPPING_L1_SUM
    assert MetricPack.index("hopping_mse_sum") == MultiTrainer._P_HOPPING_MSE_SUM
    assert MetricPack.index("hopping_cnt_sum") == MultiTrainer._P_HOPPING_CNT_SUM
    assert MetricPack.index("z_sum") == MultiTrainer._P_Z_SUM
    assert MetricPack.index("z_cnt") == MultiTrainer._P_Z_CNT
    assert MetricPack.index("cv_sum") == MultiTrainer._P_CV_SUM
    assert MetricPack.index("cv_cnt") == MultiTrainer._P_CV_CNT
    assert MetricPack.index("grad_norm_sum") == MultiTrainer._P_GRAD_NORM_SUM
    assert MetricPack.index("step_count") == MultiTrainer._P_STEP_COUNT


def test_dynamic_batch_stat_indices_match_legacy_multitrainer_constants():
    assert DynamicBatchStat.LENGTH == MultiTrainer._DB_PACK_LEN
    assert DynamicBatchStat.index("num_graphs_sum") == MultiTrainer._DB_NUM_GRAPHS_SUM
    assert DynamicBatchStat.index("cost_sum") == MultiTrainer._DB_COST_SUM
    assert DynamicBatchStat.index("num_nodes_sum") == MultiTrainer._DB_NUM_NODES_SUM
    assert DynamicBatchStat.index("num_edges_sum") == MultiTrainer._DB_NUM_EDGES_SUM
    assert DynamicBatchStat.index("max_item_cost_sum") == MultiTrainer._DB_MAX_ITEM_COST_SUM
    assert DynamicBatchStat.index("step_count") == MultiTrainer._DB_STEP_COUNT
    assert DynamicBatchStat.index("oom_skipped_count") == MultiTrainer._DB_OOM_SKIPPED_COUNT


# ---------------------------------------------------------------------------
# Declared dataclass field order must equal the wire order (no drift).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [MetricPack, DynamicBatchStat, ExpertDisplayMetric])
def test_field_order_matches_wire_order(cls):
    assert cls()._order_consistent_with_fields()


# ---------------------------------------------------------------------------
# to_tensor slot placement: a single field set to a sentinel lands in its slot
# and nowhere else.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls,slots", [
    (MetricPack, _METRIC_SLOTS),
    (DynamicBatchStat, _DB_SLOTS),
    (ExpertDisplayMetric, _DISPLAY_SLOTS),
])
def test_to_tensor_places_each_field_in_its_slot(cls, slots):
    for name, slot in slots.items():
        pack = cls(**{name: 3.5})
        vec = pack.to_tensor(dtype=torch.float64, device="cpu")
        assert vec.shape == (cls.LENGTH,)
        assert float(vec[slot].item()) == 3.5, (name, slot)
        for other in range(cls.LENGTH):
            if other != slot:
                assert float(vec[other].item()) == 0.0, (name, other)


# ---------------------------------------------------------------------------
# Round-trip: from_tensor(to_tensor) and to_tensor(from_tensor) are identity.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls,slots", [
    (MetricPack, _METRIC_SLOTS),
    (DynamicBatchStat, _DB_SLOTS),
    (ExpertDisplayMetric, _DISPLAY_SLOTS),
])
def test_round_trip_identity(cls, slots):
    # Distinct nonzero value per slot so a transposition would be caught.
    values = torch.arange(1, cls.LENGTH + 1, dtype=torch.float64)
    wrapped = cls.from_tensor(values)
    # Field access returns the right slot value.
    for name, slot in slots.items():
        assert float(getattr(wrapped, name).item()) == float(values[slot].item()), name
    # to_tensor(from_tensor(x)) == x byte-for-byte.
    rebuilt = wrapped.to_tensor(dtype=torch.float64, device="cpu")
    assert torch.equal(rebuilt, values), cls.__name__


def test_from_tensor_returns_views_not_copies():
    # Legacy pack[_P_*] reads were views into the reduced tensor; preserve that.
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
# Legacy-replica BYTE equality: build the wire tensor the OLD way (magic-int
# assignment using the retained MultiTrainer constants) and the NEW way from
# the same inputs; assert torch.equal. This is the core byte-identity proof.
# ---------------------------------------------------------------------------

def _legacy_step_pack(values, dtype, device):
    """Replica of the historical MultiTrainer._make_step_pack index writes."""
    vec = torch.zeros((MultiTrainer._PACK_LEN,), dtype=dtype, device=device)
    vec[MultiTrainer._P_LOSS_OPT_SUM] = values["loss_detached"]
    vec[MultiTrainer._P_ONSITE_WEIGHTED_SUM] = values["onsite_weighted_sum"]
    vec[MultiTrainer._P_HOPPING_WEIGHTED_SUM] = values["hopping_weighted_sum"]
    vec[MultiTrainer._P_ACTIVE_NODES_SUM] = values["active_nodes"]
    vec[MultiTrainer._P_ACTIVE_EDGES_SUM] = values["active_edges"]
    vec[MultiTrainer._P_ONSITE_L1_SUM] = values["onsite_l1_sum"]
    vec[MultiTrainer._P_ONSITE_MSE_SUM] = values["onsite_mse_sum"]
    vec[MultiTrainer._P_ONSITE_CNT_SUM] = values["onsite_cnt"]
    vec[MultiTrainer._P_HOPPING_L1_SUM] = values["hopping_l1_sum"]
    vec[MultiTrainer._P_HOPPING_MSE_SUM] = values["hopping_mse_sum"]
    vec[MultiTrainer._P_HOPPING_CNT_SUM] = values["hopping_cnt"]
    if values.get("z_cnt", 0):
        vec[MultiTrainer._P_Z_SUM] = values["z_sum"]
        vec[MultiTrainer._P_Z_CNT] = values["z_cnt"]
    if values.get("cv_cnt", 0):
        vec[MultiTrainer._P_CV_SUM] = values["cv_sum"]
        vec[MultiTrainer._P_CV_CNT] = values["cv_cnt"]
    vec[MultiTrainer._P_GRAD_NORM_SUM] = values["grad_norm"]
    vec[MultiTrainer._P_STEP_COUNT] = 1.0
    return vec


def test_metric_pack_matches_legacy_step_pack_bytes():
    values = {
        "loss_detached": 1.25,
        "onsite_weighted_sum": 2.5,
        "hopping_weighted_sum": 3.75,
        "active_nodes": 11.0,
        "active_edges": 22.0,
        "onsite_l1_sum": 0.5,
        "onsite_mse_sum": 0.25,
        "onsite_cnt": 7.0,
        "hopping_l1_sum": 0.6,
        "hopping_mse_sum": 0.36,
        "hopping_cnt": 9.0,
        "z_sum": 4.0,
        "z_cnt": 2.0,
        "cv_sum": 8.0,
        "cv_cnt": 3.0,
        "grad_norm": 1.5,
    }
    legacy = _legacy_step_pack(values, dtype=torch.float64, device="cpu")
    new = MetricPack(
        loss_opt_sum=values["loss_detached"],
        onsite_weighted_sum=values["onsite_weighted_sum"],
        hopping_weighted_sum=values["hopping_weighted_sum"],
        active_nodes_sum=values["active_nodes"],
        active_edges_sum=values["active_edges"],
        onsite_l1_sum=values["onsite_l1_sum"],
        onsite_mse_sum=values["onsite_mse_sum"],
        onsite_cnt_sum=values["onsite_cnt"],
        hopping_l1_sum=values["hopping_l1_sum"],
        hopping_mse_sum=values["hopping_mse_sum"],
        hopping_cnt_sum=values["hopping_cnt"],
        z_sum=values["z_sum"],
        z_cnt=values["z_cnt"],
        cv_sum=values["cv_sum"],
        cv_cnt=values["cv_cnt"],
        grad_norm_sum=values["grad_norm"],
        step_count=1.0,
    ).to_tensor(dtype=torch.float64, device="cpu")
    assert torch.equal(new, legacy)


def test_metric_pack_matches_legacy_when_z_and_cv_absent():
    # z/cv slots must stay zero when their counts are absent (conditional write).
    legacy = torch.zeros((MultiTrainer._PACK_LEN,), dtype=torch.float64)
    legacy[MultiTrainer._P_LOSS_OPT_SUM] = 5.0
    legacy[MultiTrainer._P_STEP_COUNT] = 1.0
    new = MetricPack(loss_opt_sum=5.0, step_count=1.0).to_tensor(
        dtype=torch.float64, device="cpu"
    )
    assert torch.equal(new, legacy)


def _legacy_dynamic_batch_pack(state, dtype, device):
    vec = torch.zeros((MultiTrainer._DB_PACK_LEN,), dtype=dtype, device=device)
    if not state:
        return vec
    vec[MultiTrainer._DB_NUM_GRAPHS_SUM] = state["batch_num_graphs"]
    vec[MultiTrainer._DB_COST_SUM] = state["batch_cost"]
    vec[MultiTrainer._DB_NUM_NODES_SUM] = state["batch_num_nodes"]
    vec[MultiTrainer._DB_NUM_EDGES_SUM] = state["batch_num_edges"]
    vec[MultiTrainer._DB_MAX_ITEM_COST_SUM] = state["batch_max_item_cost"]
    vec[MultiTrainer._DB_STEP_COUNT] = 1.0
    return vec


def test_dynamic_batch_stat_matches_legacy_pack_bytes():
    state = {
        "batch_num_graphs": 4.0,
        "batch_cost": 128.0,
        "batch_num_nodes": 40.0,
        "batch_num_edges": 96.0,
        "batch_max_item_cost": 33.0,
    }
    legacy = _legacy_dynamic_batch_pack(state, dtype=torch.float64, device="cpu")
    new = DynamicBatchStat(
        num_graphs_sum=state["batch_num_graphs"],
        cost_sum=state["batch_cost"],
        num_nodes_sum=state["batch_num_nodes"],
        num_edges_sum=state["batch_num_edges"],
        max_item_cost_sum=state["batch_max_item_cost"],
        step_count=1.0,
    ).to_tensor(dtype=torch.float64, device="cpu")
    assert torch.equal(new, legacy)


def test_dynamic_batch_stat_empty_matches_legacy_zeros():
    legacy = _legacy_dynamic_batch_pack(None, dtype=torch.float64, device="cpu")
    new = DynamicBatchStat().to_tensor(dtype=torch.float64, device="cpu")
    assert torch.equal(new, legacy)


def test_expert_display_metric_matches_legacy_stack_bytes():
    # Legacy layout: torch.stack([onsite, hopping, grad_norm, lr, nodes, edges]).
    onsite = torch.tensor(1.0, dtype=torch.float64)
    hopping = torch.tensor(2.0, dtype=torch.float64)
    grad_norm = torch.tensor(3.0, dtype=torch.float64)
    lr = torch.tensor(4.0, dtype=torch.float64)
    nodes = torch.tensor(5.0, dtype=torch.float64)
    edges = torch.tensor(6.0, dtype=torch.float64)
    legacy = torch.stack([onsite, hopping, grad_norm, lr, nodes, edges])
    new = ExpertDisplayMetric(
        expert_onsite=onsite,
        expert_hopping=hopping,
        grad_norm=grad_norm,
        lr=lr,
        active_nodes=nodes,
        active_edges=edges,
    ).to_tensor(dtype=torch.float64, device="cpu")
    assert torch.equal(new, legacy)
