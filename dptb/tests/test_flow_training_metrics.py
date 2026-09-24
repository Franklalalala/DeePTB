"""Trainer/MultiTrainer logging plumbing built on top of the Hamiltonian flows:
compatible-loss stats fast paths vs raw-batch fallback, validation() alignment
with legacy keys, per-expert payload construction, and monitor/scheduler wiring.

Pure flow numerics (prepare_batch/loss/sample, config tables, JVP) live in
test_flow_core.py instead.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dptb.data import AtomicDataDict
from dptb.nnops import trainer as trainer_module
from dptb.nnops.flow import HamiltonianPixelMeanFlow, build_hamiltonian_flow
from dptb.nnops.loss import HamilLossAbs
from dptb.nnops.multi_trainer import MultiTrainer
from dptb.nnops.trainer import Trainer
from dptb.plugins.monitor import TensorBoardMonitor, Validationer
from dptb.tests.flow_helpers import (
    BlockEndpointFallbackLoss,
    StatsCompatibleLoss,
    two_graph_batch,
)


def _scalar(value):
    if torch.is_tensor(value):
        value = value.detach()
        if value.ndim > 0:
            value = value.mean()
        return float(value.item())
    return float(value)


class _FakeBatch:
    __slices__ = {}
    __cumsum__ = {}
    __cat_dims__ = {}
    __num_nodes_list__ = []
    __data_class__ = object

    def to(self, device):
        return self


class _ValidationIdentityModel:
    def eval(self):
        return None

    def __call__(self, batch):
        out = batch.copy()
        out["node_features"] = batch["node_features"].clone()
        out["edge_features"] = batch["edge_features"].clone()
        return out


class _ValidationEndpointModel:
    def eval(self):
        return None

    def __call__(self, batch):
        out = batch.copy()
        out["node_features"] = torch.ones_like(batch["node_features"])
        out["edge_features"] = torch.full_like(batch["edge_features"], 2.0)
        return out


class _NoopTagger:
    def tag(self, name, *, it=None, expert=None, extra=""):
        class _Ctx:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        return _Ctx()


_UNUSED_MODEL = object()


# ---------------------------------------------------------------------------
# Effective per-expert LR state
# ---------------------------------------------------------------------------


def test_single_trainer_effective_expert_lr_state_uses_global_optimizer_lr():
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([param], lr=0.0125)
    state = {}

    Trainer._add_effective_expert_lr_state(state, optimizer=optimizer, num_experts=2)

    assert state["expert_0_lr"] == pytest.approx(0.0125)
    assert state["expert_1_lr"] == pytest.approx(0.0125)


# ---------------------------------------------------------------------------
# HamilLossAbs compatible-stats fast path matches its own forward() semantics
# ---------------------------------------------------------------------------


class _LossIDP:
    def __init__(self):
        self.mask_to_nrme = torch.tensor([[True, True, False], [True, False, True]])
        self.mask_to_erme = torch.tensor([[True, False, True], [False, True, True]])


def _masked_stats(diff, mask):
    mask_f = mask.to(dtype=diff.dtype)
    return {
        "l1_sum": (diff.abs() * mask_f).sum(),
        "mse_sum": (diff.square() * mask_f).sum(),
        "count": mask_f.sum().to(dtype=diff.dtype),
    }


@pytest.mark.parametrize(
    "loss_kwargs",
    [
        {"element_average": False, "z_loss_coef": 0.2},
        {"element_average": True, "z_loss_coef": 0.2},
        {"onsite_boost": True, "onsite_boost_steps": 100, "onsite_boost_max": 3.0},
    ],
)
def test_hamil_abs_compatible_stats_match_forward_semantics(loss_kwargs):
    idp = _LossIDP()
    lossfunc = HamilLossAbs(idp=idp, dtype=torch.float64, **loss_kwargs)
    pred = {
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor([0, 1]),
        AtomicDataDict.EDGE_TYPE_KEY: torch.tensor([0, 1]),
        AtomicDataDict.NODE_FEATURES_KEY: torch.tensor(
            [[1.0, -2.0, 9.0], [3.0, 8.0, -4.0]], dtype=torch.float64
        ),
        AtomicDataDict.EDGE_FEATURES_KEY: torch.tensor(
            [[5.0, 7.0, -6.0], [11.0, -8.0, 2.0]], dtype=torch.float64
        ),
        "mean_max_prob": torch.tensor(0.25, dtype=torch.float64),
        "global_step": 25,
    }
    ref = {
        AtomicDataDict.NODE_FEATURES_KEY: torch.tensor(
            [[0.5, -1.0, 0.0], [1.0, 0.0, -1.0]], dtype=torch.float64
        ),
        AtomicDataDict.EDGE_FEATURES_KEY: torch.tensor(
            [[3.0, 0.0, -2.0], [0.0, -5.0, 3.0]], dtype=torch.float64
        ),
    }

    forward_total = lossfunc(pred, ref)
    forward_onsite = lossfunc.last_onsite_loss.detach().clone()
    forward_hopping = lossfunc.last_hopping_loss.detach().clone()
    node_mask = idp.mask_to_nrme[pred[AtomicDataDict.ATOM_TYPE_KEY].flatten()]
    edge_mask = idp.mask_to_erme[pred[AtomicDataDict.EDGE_TYPE_KEY].flatten()]
    node_stats = _masked_stats(
        pred[AtomicDataDict.NODE_FEATURES_KEY] - ref[AtomicDataDict.NODE_FEATURES_KEY], node_mask
    )
    edge_stats = _masked_stats(
        pred[AtomicDataDict.EDGE_FEATURES_KEY] - ref[AtomicDataDict.EDGE_FEATURES_KEY], edge_mask
    )

    stats_total, stats_onsite, stats_hopping = lossfunc.compatible_loss_from_stats(
        onsite_l1_sum=node_stats["l1_sum"],
        onsite_mse_sum=node_stats["mse_sum"],
        onsite_count=node_stats["count"],
        hopping_l1_sum=edge_stats["l1_sum"],
        hopping_mse_sum=edge_stats["mse_sum"],
        hopping_count=edge_stats["count"],
        z_loss=pred["mean_max_prob"],
        global_step=pred["global_step"],
    )

    torch.testing.assert_close(stats_total, forward_total.detach())
    torch.testing.assert_close(stats_onsite, forward_onsite)
    torch.testing.assert_close(stats_hopping, forward_hopping)


def test_hamil_abs_compatible_loss_projects_raw_uureal_predictions_to_compressed_targets():
    from dptb.data import _keys
    from dptb.tests.flow_helpers import FakeIDP

    device = torch.device("cpu")
    dtype = torch.float64
    raw_mask = torch.tensor([1, 1, 0, 1, 0, 1, 0, 0], device=device, dtype=torch.bool)
    idp = FakeIDP(device=device)
    idp.mask_uureal = raw_mask
    idp.mask_to_nrme = raw_mask.expand(2, -1).clone()
    idp.mask_to_erme = raw_mask.expand(2, -1).clone()
    lossfunc = HamilLossAbs(idp=idp)

    node_target = torch.arange(12, device=device, dtype=dtype).reshape(3, 4) / 10.0
    edge_target = torch.arange(16, device=device, dtype=dtype).reshape(4, 4) / 10.0
    node_pred_raw = torch.zeros(3, raw_mask.numel(), device=device, dtype=dtype)
    edge_pred_raw = torch.zeros(4, raw_mask.numel(), device=device, dtype=dtype)
    node_pred_raw[:, raw_mask] = node_target + 0.5
    edge_pred_raw[:, raw_mask] = edge_target + 0.25

    pred_data = {
        _keys.NODE_FEATURES_KEY: node_pred_raw,
        _keys.EDGE_FEATURES_KEY: edge_pred_raw,
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor([0, 1, 0], device=device, dtype=torch.long),
        AtomicDataDict.EDGE_TYPE_KEY: torch.tensor([0, 1, 0, 1], device=device, dtype=torch.long),
    }
    ref_data = {_keys.NODE_FEATURES_KEY: node_target, _keys.EDGE_FEATURES_KEY: edge_target}

    state = Trainer._compatible_loss_state(lossfunc, pred_data, ref_data, prefix="train")

    assert state["train_onsite_loss"].item() == pytest.approx(0.5)
    assert state["train_hopping_loss"].item() == pytest.approx(0.25)
    assert state["train_loss"].item() == pytest.approx(0.375)


# ---------------------------------------------------------------------------
# Trainer._compatible_loss_state: no-grad, side effects, legacy-prefix mapping
# ---------------------------------------------------------------------------


class _ComponentLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.called_with_grad_enabled = None
        self.last_onsite_loss = torch.tensor(123.0)
        self.last_hopping_loss = torch.tensor(456.0)

    def forward(self, pred, ref):
        self.called_with_grad_enabled = torch.is_grad_enabled()
        onsite = (pred["node_features"] - ref["node_features"]).abs().mean()
        hopping = (pred["edge_features"] - ref["edge_features"]).abs().mean()
        self.last_onsite_loss = onsite.detach()
        self.last_hopping_loss = hopping.detach()
        return 0.5 * (onsite + hopping)


class _DistinctEndpointLoss(torch.nn.Module):
    endpoint_metric_space = "block"

    def __init__(self):
        super().__init__()
        self.last_endpoint_loss = torch.tensor(99.0)
        self.last_endpoint_metric_space = "saved"
        self.last_onsite_loss = torch.tensor(88.0)
        self.last_hopping_loss = torch.tensor(77.0)

    def forward(self, pred, ref):
        self.last_endpoint_loss = torch.tensor(2.0)
        self.last_endpoint_metric_space = "block"
        self.last_onsite_loss = torch.tensor(1.0)
        self.last_hopping_loss = torch.tensor(3.0)
        return torch.tensor(20.0)


def _pred_ref():
    pred = {
        "node_features": torch.tensor([[1.0], [3.0]], requires_grad=True),
        "edge_features": torch.tensor([[2.0]], requires_grad=True),
    }
    ref = {"node_features": torch.zeros(2, 1), "edge_features": torch.zeros(1, 1)}
    return pred, ref


@pytest.mark.parametrize(
    "prefix,legacy_prefix",
    [
        ("train_compatible", None),
        ("train_compatible", "train"),
        ("validation_compatible_euler_1", "validation"),
    ],
    ids=["no_legacy_prefix", "explicit_legacy_mapping", "maps_validation_clean_legacy_loss"],
)
def test_flow_compatible_loss_state_uses_no_grad_and_restores_side_effects(prefix, legacy_prefix):
    lossfunc = _ComponentLoss()
    pred, ref = _pred_ref()

    state = Trainer._compatible_loss_state(
        lossfunc, pred, ref, prefix=prefix, legacy_prefix=legacy_prefix
    )

    assert lossfunc.called_with_grad_enabled is False
    assert state[f"{prefix}_loss"].requires_grad is False
    assert state[f"{prefix}_onsite_loss"].item() == pytest.approx(2.0)
    assert state[f"{prefix}_hopping_loss"].item() == pytest.approx(2.0)
    assert lossfunc.last_onsite_loss.item() == pytest.approx(123.0)
    assert lossfunc.last_hopping_loss.item() == pytest.approx(456.0)
    if legacy_prefix is None:
        assert "train_onsite_loss" not in state
        assert "train_hopping_loss" not in state
    else:
        assert state[f"{legacy_prefix}_onsite_loss"].item() == pytest.approx(2.0)
        assert state[f"{legacy_prefix}_hopping_loss"].item() == pytest.approx(2.0)
        if legacy_prefix == "validation":
            assert state[f"{legacy_prefix}_loss"].item() == pytest.approx(2.0)


@pytest.mark.parametrize("via_multitrainer_full_forward", [False, True])
def test_compatible_forward_uses_endpoint_total_not_optimization_total(via_multitrainer_full_forward):
    lossfunc = _DistinctEndpointLoss()

    if via_multitrainer_full_forward:
        trainer = object.__new__(MultiTrainer)
        trainer.iter = 1
        trainer.model = lambda batch: dict(batch)
        assert trainer._run_full_batch_loss({}, {}, lossfunc).item() == pytest.approx(2.0)
        return

    state = Trainer._compatible_loss_state(
        lossfunc, {}, {}, prefix="validation_compatible_euler_1", legacy_prefix="validation"
    )

    assert state["validation_loss"].item() == pytest.approx(2.0)
    assert state["validation_onsite_loss"].item() == pytest.approx(1.0)
    assert state["validation_hopping_loss"].item() == pytest.approx(3.0)
    assert state["validation_compatible_euler_1_loss_opt"].item() == pytest.approx(20.0)
    assert lossfunc.last_endpoint_loss.item() == pytest.approx(99.0)
    assert lossfunc.last_endpoint_metric_space == "saved"


def _compatible_clean_stats():
    return {
        "_compatible_clean_stats": {
            "onsite_l1_sum": torch.tensor(4.0),
            "onsite_mse_sum": torch.tensor(10.0),
            "onsite_count": torch.tensor(2.0),
            "hopping_l1_sum": torch.tensor(3.0),
            "hopping_mse_sum": torch.tensor(9.0),
            "hopping_count": torch.tensor(3.0),
        },
        "mean_max_prob": torch.tensor(0.75),
        "expert_load_cv": torch.tensor(0.25),
    }


def test_compatible_loss_state_from_flow_stats_fails_fast_on_opt_in_mismatch():
    from dptb.nnops.blockwise_nextham_loss import HamilBlockwiseNexTHamLoss

    lossfunc = HamilBlockwiseNexTHamLoss(basis={"H": "1s"})
    assert lossfunc.endpoint_metric_space == "block"
    flow_state = _compatible_clean_stats()
    flow_state["_compatible_clean_stats"]["metric_space"] = "rme"

    # Default stays a silent None (preserves the cross-representation fallback
    # contract exercised by MultiTrainer's FlowObjective / _run_one_expert_loss).
    assert Trainer._compatible_loss_state_from_flow_stats(
        lossfunc, flow_state, source_prefix="train", prefix="train_compatible",
        legacy_prefix="train", fail_on_metric_space_mismatch=False,
    ) is None

    with pytest.raises(ValueError, match=r"metric_space='rme'.*endpoint_metric_space='block'"):
        Trainer._compatible_loss_state_from_flow_stats(
            lossfunc, flow_state, source_prefix="train", prefix="train_compatible",
            legacy_prefix="train", fail_on_metric_space_mismatch=True,
        )

    # A genuine match must never raise regardless of the flag: opting in to
    # fail-fast only changes the mismatch branch.
    flow_state["_compatible_clean_stats"]["metric_space"] = "block"
    state = Trainer._compatible_loss_state_from_flow_stats(
        lossfunc, flow_state, source_prefix="train", prefix="train_compatible",
        legacy_prefix="train", fail_on_metric_space_mismatch=True,
    )
    assert state is not None
    assert state["train_loss"].item() == pytest.approx(
        0.5 * (state["train_onsite_loss"] + state["train_hopping_loss"]).item()
    )


class _BlockOdeMismatchFlow:
    """block_ode=True flow whose euler sample publishes a mismatched label.

    validation_ode_steps has a single value that is NOT 1, exercising the
    num_steps != 1 branch: pre-fix, that branch had no metric-space safety net
    and crashed inside Trainer._accumulate_metric_state with an uninformative
    AttributeError instead of the clear metric-space-mismatch ValueError.
    """

    enabled = True
    model_in_loss = False
    block_ode = True
    log_validation_compatible_loss = True
    compatible_loss_to_legacy_keys = True
    validation_ode_steps = (3,)
    log_validation_random_t_loss = False
    log_validation_t0_loss = False
    log_validation_flow_euler_loss = False

    def _num_graphs(self, batch):
        return 1

    def prepare_batch(self, original_batch, batch_for_loss, t=None, **kwargs):
        return original_batch.copy(), batch_for_loss.copy(), object()

    def sample(self, model, batch, *, num_steps):
        return model(batch)

    def compatible_loss_on_sample(self, sampled, flow_ref, flow_ctx):
        state = {
            "_compatible_clean_stats": {
                "onsite_l1_sum": torch.tensor(1.0),
                "onsite_mse_sum": torch.tensor(1.0),
                "onsite_count": torch.tensor(1.0),
                "hopping_l1_sum": torch.tensor(1.0),
                "hopping_mse_sum": torch.tensor(1.0),
                "hopping_count": torch.tensor(1.0),
                "metric_space": "rme",
            }
        }
        return torch.tensor(0.0), state


class _BlockLabelCriterion:
    endpoint_metric_space = "block"


def test_validation_block_ode_metric_space_mismatch_fails_fast(monkeypatch):
    trainer = object.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.dtype = torch.float32
    trainer.model = _ValidationIdentityModel()
    trainer.flow_cfm = _BlockOdeMismatchFlow()
    trainer.validation_loader = [_FakeBatch()]
    trainer.validation_lossfunc = _BlockLabelCriterion()
    trainer.iter = 5

    monkeypatch.setattr(
        trainer_module.AtomicData, "to_AtomicDataDict", lambda batch: two_graph_batch()
    )

    with pytest.raises(ValueError, match=r"metric_space='rme'.*endpoint_metric_space='block'"):
        trainer.validation(fast=True)


def test_flow_stats_fast_path_preserves_compatible_and_legacy_semantics():
    lossfunc = StatsCompatibleLoss()
    state = Trainer._compatible_loss_state_from_flow_stats(
        lossfunc, _compatible_clean_stats(), source_prefix="train", prefix="train_compatible",
        legacy_prefix="train", global_step=17,
    )

    onsite = 0.5 * (2.0 + (10.0 / 2.0) ** 0.5)
    hopping = 0.5 * (1.0 + 3.0 ** 0.5)
    total = 0.5 * (onsite + hopping)
    assert lossfunc.forward_calls == 0
    assert lossfunc.stats_calls == 1
    assert state["train_compatible_loss"].item() == pytest.approx(total)
    assert state["train_loss"].item() == pytest.approx(total)
    assert state["train_onsite_loss"].item() == pytest.approx(onsite)
    assert state["train_hopping_loss"].item() == pytest.approx(hopping)


# ---------------------------------------------------------------------------
# Trainer.validation(): euler-only / multi-step compatible legacy-key mapping
# ---------------------------------------------------------------------------


class _EulerOnlyValidationFlow:
    enabled = True
    model_in_loss = False
    log_validation_compatible_loss = True
    compatible_loss_to_legacy_keys = True
    validation_ode_steps = (1,)
    log_validation_random_t_loss = False
    log_validation_t0_loss = False
    log_validation_flow_euler_loss = False

    def __init__(self):
        self.sample_calls = 0

    def _num_graphs(self, batch):
        return 1

    def prepare_batch(self, *args, **kwargs):
        raise AssertionError("Euler-only validation should not prepare random-t/t0 batches")

    def loss(self, *args, **kwargs):
        raise AssertionError("Euler-only validation should not compute flow validation loss")

    def sample(self, model, batch, *, num_steps):
        assert num_steps == 1
        self.sample_calls += 1
        return model(batch)


def test_validation_euler_only_compatible_maps_to_legacy_loss(monkeypatch):
    trainer = object.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.dtype = torch.float32
    trainer.model = _ValidationEndpointModel()
    trainer.flow_cfm = _EulerOnlyValidationFlow()
    trainer.validation_loader = [_FakeBatch()]
    trainer.validation_lossfunc = _ComponentLoss()
    trainer.iter = 4

    monkeypatch.setattr(
        trainer_module.AtomicData, "to_AtomicDataDict", lambda batch: two_graph_batch()
    )

    loss = trainer.validation(fast=True)

    # validation() returns the compatible legacy loss when it exists, so direct
    # callers see the same aligned scalar Validationer reports.
    assert loss.item() == pytest.approx(1.5)
    assert trainer.flow_cfm.sample_calls == 1
    assert "validation_flow_random_t_loss" not in trainer._last_flow_validation_state
    assert "validation_flow_t0_loss" not in trainer._last_flow_validation_state
    assert "validation_flow_euler_1_loss" not in trainer._last_flow_validation_state
    state = trainer._last_flow_validation_state
    assert state["validation_compatible_euler_1_loss"].item() == pytest.approx(1.5)
    assert state["validation_loss"].item() == pytest.approx(1.5)
    assert state["validation_onsite_loss"].item() == pytest.approx(1.0)
    assert state["validation_hopping_loss"].item() == pytest.approx(2.0)


class _MultiEulerValidationFlow:
    enabled = True
    model_in_loss = False
    log_validation_compatible_loss = True
    compatible_loss_to_legacy_keys = True
    validation_ode_steps = (1, 3)

    def __init__(self):
        self.sample_calls = []

    def prepare_batch(self, *args, **kwargs):
        raise AssertionError("MultiTrainer compatible validation should not use random-t batches")

    def loss(self, *args, **kwargs):
        raise AssertionError("MultiTrainer compatible validation should not use flow loss")

    def sample(self, model, batch, *, num_steps):
        assert "expert_node_mask" in batch
        assert "expert_edge_mask" in batch
        self.sample_calls.append(int(num_steps))
        out = batch.copy()
        out["node_features"] = torch.full_like(batch["node_features"], float(num_steps))
        out["edge_features"] = torch.full_like(batch["edge_features"], float(2 * num_steps))
        return out


class _StatsForwardLoss(torch.nn.Module):
    """A criterion whose forward() recomputes stats from pred/ref directly.

    Unlike StatsCompatibleLoss (whose forward() must never run because the
    fast stats path is what should be used), the euler-sample validation path
    genuinely calls the criterion on the sampled endpoint, so this stub's
    forward() must do real work.
    """

    def __init__(self):
        super().__init__()
        self.last_onsite_loss = None
        self.last_hopping_loss = None

    def forward(self, pred, ref):
        node_mask = (
            pred.get("expert_node_mask", torch.ones(pred["node_features"].shape[0], dtype=torch.bool))
            .to(dtype=pred["node_features"].dtype)
            .unsqueeze(-1)
        )
        edge_mask = (
            pred.get("expert_edge_mask", torch.ones(pred["edge_features"].shape[0], dtype=torch.bool))
            .to(dtype=pred["edge_features"].dtype)
            .unsqueeze(-1)
        )
        node_diff = (pred["node_features"] - ref["node_features"]) * node_mask
        edge_diff = (pred["edge_features"] - ref["edge_features"]) * edge_mask
        onsite_l1_sum = node_diff.abs().sum().detach()
        onsite_mse_sum = node_diff.square().sum().detach()
        onsite_count = node_mask.sum().detach()
        hopping_l1_sum = edge_diff.abs().sum().detach()
        hopping_mse_sum = edge_diff.square().sum().detach()
        hopping_count = edge_mask.sum().detach()
        onsite = 0.5 * (
            onsite_l1_sum / onsite_count.clamp_min(1.0)
            + torch.sqrt(onsite_mse_sum / onsite_count.clamp_min(1.0) + 1e-12)
        )
        hopping = 0.5 * (
            hopping_l1_sum / hopping_count.clamp_min(1.0)
            + torch.sqrt(hopping_mse_sum / hopping_count.clamp_min(1.0) + 1e-12)
        )
        self.last_onsite_loss = onsite.detach()
        self.last_hopping_loss = hopping.detach()
        return 0.5 * (onsite + hopping)


def test_multitrainer_validation_uses_euler_sample_for_compatible_legacy_loss():
    trainer = object.__new__(MultiTrainer)
    trainer.iter = 11
    trainer.dtype = torch.float32
    trainer.device = torch.device("cpu")
    trainer._tagger = _NoopTagger()
    trainer.model = _ValidationIdentityModel()
    trainer.flow_cfm = _MultiEulerValidationFlow()
    trainer.validation_loader = [_FakeBatch()]
    trainer.validation_lossfunc = _StatsForwardLoss()
    trainer.distributed_expert = True
    trainer.local_expert_idx = 0
    trainer.distance_ranges = [(0.0, 1.0)]
    trainer.world_size = 1
    trainer.endpoint_loss_mode = "reduce"
    trainer._prepare_batch_bundle = lambda batch, with_lengths=True: (two_graph_batch(), {})
    trainer._prepare_expert_masks = lambda batch, range_dis, expert_idx: (
        torch.ones(batch["edge_features"].shape[0], dtype=torch.bool),
        torch.ones(batch["node_features"].shape[0], dtype=torch.bool),
    )
    trainer._all_reduce_ = lambda tensor, name=None: tensor

    loss = trainer.validation(fast=True)

    assert trainer.flow_cfm.sample_calls == [1, 3]
    assert loss.item() == pytest.approx(1.5)
    state = trainer._last_flow_validation_state
    assert state["validation_loss"].item() == pytest.approx(1.5)
    assert state["validation_onsite_loss"].item() == pytest.approx(1.0)
    assert state["validation_hopping_loss"].item() == pytest.approx(2.0)
    assert state["validation_compatible_euler_1_loss"].item() == pytest.approx(1.5)
    assert state["validation_compatible_euler_3_loss"].item() == pytest.approx(4.5)


# ---------------------------------------------------------------------------
# MultiTrainer per-expert payload construction: flow applied before the model
# ---------------------------------------------------------------------------


class _FlowPreparedModel:
    def __init__(self):
        self.seen = None

    def __call__(self, batch):
        self.seen = batch.copy()
        out = batch.copy()
        out["node_features"] = batch["node_h0"].clone()
        out["edge_features"] = batch["edge_h0"].clone()
        return out


class _PreparedFlow:
    enabled = True
    model_in_loss = False
    log_train_compatible_loss = False
    compatible_loss_to_legacy_keys = True

    def __init__(self):
        self.prepare_called = False
        self.loss_called = False

    def prepare_batch(self, batch, ref_batch):
        self.prepare_called = True
        out = batch.copy()
        ref = ref_batch.copy()
        out["node_h0"] = batch["node_h0"] + 10.0
        out["edge_h0"] = batch["edge_h0"] + 20.0
        return out, ref, object()

    def loss(self, pred, ref, ctx):
        self.loss_called = True
        assert torch.equal(pred["node_features"], pred["node_h0"])
        assert torch.equal(pred["edge_features"], pred["edge_h0"])
        loss = pred["node_features"].sum() * 0.0 + torch.tensor(5.0)
        return loss, {
            "train_flow_loss": torch.tensor(5.0),
            "train_onsite_loss": torch.tensor(2.0),
            "train_hopping_loss": torch.tensor(3.0),
        }


class _PreparedFlowWithStats(_PreparedFlow):
    log_train_compatible_loss = True

    def loss(self, pred, ref, ctx):
        self.loss_called = True
        return pred["node_features"].sum() * 0.0 + torch.tensor(5.0), {
            "train_flow_loss": torch.tensor(5.0),
            **_compatible_clean_stats(),
        }


class _PreparedFlowWithRMEMetricStats(_PreparedFlow):
    def loss(self, pred, ref, ctx):
        state = _compatible_clean_stats()
        state["_compatible_clean_stats"]["metric_space"] = "rme"
        return pred["node_features"].sum() * 0.0 + torch.tensor(5.0), {
            "train_flow_loss": torch.tensor(5.0),
            **state,
        }


def test_multitrainer_expert_payload_applies_flow_before_model():
    trainer = object.__new__(MultiTrainer)
    trainer.iter = 1
    trainer.dtype = torch.float32
    trainer.device = torch.device("cpu")
    trainer._tagger = _NoopTagger()
    trainer.flow_cfm = _PreparedFlow()
    trainer.model = _FlowPreparedModel()
    trainer._prepare_expert_masks = lambda batch, range_dis, expert_idx: (
        torch.ones(batch["edge_h0"].shape[0], dtype=torch.bool),
        torch.ones(batch["node_h0"].shape[0], dtype=torch.bool),
    )

    batch = two_graph_batch()
    result = trainer._run_one_expert_loss(
        batch, batch_info={}, criterion=_ComponentLoss(), expert_idx=0,
        range_dis=(0.0, 1.0), capture_metrics=True,
    )

    assert trainer.flow_cfm.prepare_called
    assert trainer.flow_cfm.loss_called
    assert torch.equal(trainer.model.seen["node_h0"], batch["node_h0"] + 10.0)
    assert torch.equal(trainer.model.seen["edge_h0"], batch["edge_h0"] + 20.0)
    assert result["loss"].item() == pytest.approx(5.0)
    assert result["onsite"].item() == pytest.approx(10.0)
    assert result["hopping"].item() == pytest.approx(20.0)


class _ModelInLossFlow:
    enabled = True
    model_in_loss = True
    apply_to_reference = False
    log_train_compatible_loss = True
    compatible_loss_to_legacy_keys = True

    def loss_with_model(self, model, batch, batch_for_loss):
        assert model is _UNUSED_MODEL
        assert batch is not batch_for_loss
        return torch.tensor(7.0, requires_grad=True), {"train_flow_loss": torch.tensor(7.0)}


class _ModelInLossFlowWithStats(_ModelInLossFlow):
    def loss_with_model(self, model, batch, batch_for_loss):
        loss, state = super().loss_with_model(model, batch, batch_for_loss)
        state.update(_compatible_clean_stats())
        return loss, state


def _case_non_display_step_uses_stats_fast_path():
    trainer = object.__new__(MultiTrainer)
    trainer.iter = 2
    trainer.dtype = torch.float32
    trainer.device = torch.device("cpu")
    trainer._tagger = _NoopTagger()
    trainer.flow_cfm = _PreparedFlowWithStats()
    trainer.model = _FlowPreparedModel()
    trainer._prepare_expert_masks = lambda batch, range_dis, expert_idx: (
        torch.ones(batch["edge_h0"].shape[0], dtype=torch.bool),
        torch.ones(batch["node_h0"].shape[0], dtype=torch.bool),
    )
    lossfunc = StatsCompatibleLoss()

    result = trainer._run_one_expert_loss(
        two_graph_batch(), batch_info={}, criterion=lossfunc, expert_idx=0,
        range_dis=(0.0, 1.0), capture_metrics=True,
    )

    assert lossfunc.forward_calls == 0
    assert lossfunc.stats_calls == 1
    assert result["last_onsite_count"].item() == pytest.approx(2.0)
    assert result["last_hopping_count"].item() == pytest.approx(3.0)
    assert result["onsite"].item() > 0.0
    assert result["hopping"].item() > 0.0


def _case_cross_space_stats_fall_back_to_raw_batch_forward():
    trainer = object.__new__(MultiTrainer)
    trainer.iter = 2
    trainer.dtype = torch.float32
    trainer.device = torch.device("cpu")
    trainer._tagger = _NoopTagger()
    trainer.flow_cfm = _PreparedFlowWithRMEMetricStats()
    trainer.model = _FlowPreparedModel()
    trainer._prepare_expert_masks = lambda batch, range_dis, expert_idx: (
        torch.ones(batch["edge_h0"].shape[0], dtype=torch.bool),
        torch.ones(batch["node_h0"].shape[0], dtype=torch.bool),
    )
    lossfunc = BlockEndpointFallbackLoss()

    result = trainer._run_one_expert_loss(
        two_graph_batch(), batch_info={}, criterion=lossfunc, expert_idx=0,
        range_dis=(0.0, 1.0), capture_metrics=True,
    )

    assert lossfunc.stats_calls == 0
    assert lossfunc.forward_calls == 1
    assert result["onsite"].item() == pytest.approx(10.0)
    assert result["hopping"].item() == pytest.approx(20.0)
    assert result["last_onsite_l1_sum"].item() == pytest.approx(30.0)
    assert result["last_hopping_l1_sum"].item() == pytest.approx(40.0)


def _case_model_in_loss_uses_flow_stats_for_compatible_metrics():
    trainer = object.__new__(MultiTrainer)
    trainer.iter = 2
    trainer.dtype = torch.float32
    trainer.device = torch.device("cpu")
    trainer._tagger = _NoopTagger()
    trainer.flow_cfm = _ModelInLossFlowWithStats()
    trainer.model = _UNUSED_MODEL
    trainer._prepare_expert_masks = lambda batch, range_dis, expert_idx: (
        torch.ones(batch["edge_h0"].shape[0], dtype=torch.bool),
        torch.ones(batch["node_h0"].shape[0], dtype=torch.bool),
    )
    lossfunc = StatsCompatibleLoss()

    result = trainer._run_one_expert_loss(
        two_graph_batch(), batch_info={}, criterion=lossfunc, expert_idx=0,
        range_dis=(0.0, 1.0), capture_metrics=True,
    )

    onsite = 0.5 * (2.0 + (10.0 / 2.0) ** 0.5)
    hopping = 0.5 * (1.0 + 3.0 ** 0.5)
    assert result["loss"].item() == pytest.approx(7.0)
    assert lossfunc.forward_calls == 0
    assert lossfunc.stats_calls == 1
    assert result["onsite"].item() == pytest.approx(onsite)
    assert result["hopping"].item() == pytest.approx(hopping)
    assert result["last_onsite_count"].item() == pytest.approx(2.0)
    assert result["last_hopping_count"].item() == pytest.approx(3.0)


_RUN_ONE_EXPERT_LOSS_CASES = [
    ("non_display_step_uses_stats_fast_path", _case_non_display_step_uses_stats_fast_path),
    ("cross_space_stats_fall_back_to_raw_batch_forward",
     _case_cross_space_stats_fall_back_to_raw_batch_forward),
    ("model_in_loss_uses_flow_stats_for_compatible_metrics",
     _case_model_in_loss_uses_flow_stats_for_compatible_metrics),
]


@pytest.mark.parametrize(
    "run_case",
    [case for _, case in _RUN_ONE_EXPERT_LOSS_CASES],
    ids=[name for name, _ in _RUN_ONE_EXPERT_LOSS_CASES],
)
def test_multitrainer_run_one_expert_loss_picks_stats_or_fallback_correctly(run_case):
    run_case()


def test_multitrainer_reference_changes_opt_only_and_obeys_flow_scope():
    trainer = object.__new__(MultiTrainer)
    trainer.dtype = torch.float32
    trainer.device = torch.device("cpu")
    trainer.flow_cfm = SimpleNamespace(enabled=True, apply_to_reference=False)
    trainer.reference_lossfunc = object()

    main = {
        "loss": torch.tensor(2.0),
        "active_nodes": torch.tensor(3.0),
        "active_edges": torch.tensor(2.0),
        "onsite": torch.tensor(1.0),
        "hopping": torch.tensor(4.0),
        "last_onsite_l1_sum": torch.tensor(3.0),
        "last_onsite_mse_sum": torch.tensor(3.0),
        "last_onsite_count": torch.tensor(3.0),
        "last_hopping_l1_sum": torch.tensor(8.0),
        "last_hopping_mse_sum": torch.tensor(32.0),
        "last_hopping_count": torch.tensor(2.0),
        "z_loss": None,
        "expert_load_cv": None,
    }
    reference = dict(main)
    reference.update(
        loss=torch.tensor(3.0), active_nodes=torch.tensor(30.0), active_edges=torch.tensor(20.0),
        onsite=torch.tensor(10.0), hopping=torch.tensor(40.0),
    )

    def fake_run(**kwargs):
        return reference if kwargs["batch_dict"]["kind"] == "reference" else main

    trainer._run_one_expert_loss = fake_run
    payload = trainer._build_train_payload(
        {"kind": "main"}, batch_info={}, expert_idx=0, range_dis=(0.0, 1.0),
        ref_batch_dict={"kind": "reference"}, ref_batch_info={}, criterion=object(),
    )

    # Only the main batch's optimization loss is used; the reference batch only
    # ever contributes through its own separately-scoped reference_lossfunc path
    # (exercised elsewhere), never mixed into the main payload's numbers.
    assert payload["loss"].item() == pytest.approx(5.0)
    assert payload["active_nodes"].item() == pytest.approx(3.0)
    assert payload["onsite_l1_sum"].item() == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Endpoint stats feed the compatible-loss reducer with the right pack fields
# ---------------------------------------------------------------------------


class _NonUniformEndpoint(torch.nn.Module):
    def forward(self, data):
        data = data.copy()
        data["node_features"] = torch.tensor(
            [[1.0], [2.0], [0.0]], device=data["node_h0"].device, dtype=data["node_h0"].dtype
        )
        data["edge_features"] = torch.tensor(
            [[1.0], [4.0]], device=data["edge_h0"].device, dtype=data["edge_h0"].dtype
        )
        return data


def test_pixel_meanflow_train_endpoint_stats_feed_compatible_reducer():
    trainer = object.__new__(MultiTrainer)
    trainer.iter = 2
    trainer.dtype = torch.float32
    trainer.device = torch.device("cpu")
    trainer._tagger = _NoopTagger()
    trainer.endpoint_loss_mode = "reduce"
    trainer.flow_cfm = HamiltonianPixelMeanFlow(
        {
            "enabled": True, "objective": "pixel_meanflow", "mode": "residual", "prior": "zero",
            "strict_h0": True,
            "meanflow": {"aux_endpoint_weight": 0.0, "aux_boundary_v_weight": 0.0, "fd_eps": 1.0e-4},
        }
    )
    trainer.model = _NonUniformEndpoint()
    trainer._prepare_expert_masks = lambda batch, range_dis, expert_idx: (
        torch.ones(batch["edge_h0"].shape[0], dtype=torch.bool),
        torch.ones(batch["node_h0"].shape[0], dtype=torch.bool),
    )
    lossfunc = StatsCompatibleLoss()

    payload = trainer._build_train_payload(
        two_graph_batch(), batch_info={}, criterion=lossfunc, expert_idx=0, range_dis=(0.0, 1.0)
    )
    pack = trainer._make_step_pack(payload)
    stats_calls_before_reduce = lossfunc.stats_calls
    state = trainer._compute_compatible_state_from_pack(pack, criterion=lossfunc, prefix="train")

    onsite = 0.5 * (1.0 + (5.0 / 3.0) ** 0.5)
    hopping = 0.5 * (2.5 + (17.0 / 2.0) ** 0.5)
    total = 0.5 * (onsite + hopping)
    assert payload["onsite_l1_sum"].item() == pytest.approx(3.0)
    assert payload["onsite_mse_sum"].item() == pytest.approx(5.0)
    assert payload["onsite_cnt"].item() == pytest.approx(3.0)
    assert payload["hopping_l1_sum"].item() == pytest.approx(5.0)
    assert payload["hopping_mse_sum"].item() == pytest.approx(17.0)
    assert payload["hopping_cnt"].item() == pytest.approx(2.0)
    assert state["train_onsite_loss"].item() == pytest.approx(onsite)
    assert state["train_hopping_loss"].item() == pytest.approx(hopping)
    assert state["train_loss"].item() == pytest.approx(total)
    assert lossfunc.forward_calls == 0
    assert stats_calls_before_reduce == 1
    assert lossfunc.stats_calls == stats_calls_before_reduce + 1


# ---------------------------------------------------------------------------
# Trainer/MultiTrainer init-time endpoint-contract guard
# ---------------------------------------------------------------------------


def _case_train_criterion_mismatch(trainer_cls):
    trainer = object.__new__(trainer_cls)
    trainer.flow_cfm = HamiltonianPixelMeanFlow({"enabled": True, "objective": "pixel_meanflow"})
    trainer.train_lossfunc = BlockEndpointFallbackLoss()
    trainer.use_reference = False
    with pytest.raises(ValueError, match="trainer initialization"):
        trainer._assert_model_in_loss_endpoint_contract()


def _case_validation_criterion_mismatch():
    trainer = object.__new__(Trainer)
    trainer.flow_cfm = HamiltonianPixelMeanFlow({"enabled": True, "objective": "pixel_meanflow"})
    trainer.train_lossfunc = SimpleNamespace(endpoint_metric_space="rme")
    trainer.validation_lossfunc = BlockEndpointFallbackLoss()
    trainer.use_validation = True
    trainer.use_reference = False
    with pytest.raises(ValueError, match=r"metric-space mismatch.*validation criterion"):
        trainer._assert_model_in_loss_endpoint_contract()


@pytest.mark.parametrize(
    "run_case",
    [
        lambda: _case_train_criterion_mismatch(Trainer),
        lambda: _case_train_criterion_mismatch(MultiTrainer),
        _case_validation_criterion_mismatch,
    ],
    ids=["train_criterion_mismatch_single", "train_criterion_mismatch_multi",
         "validation_criterion_mismatch"],
)
def test_endpoint_contract_guard_fails_fast_at_init(run_case):
    run_case()


# ---------------------------------------------------------------------------
# Validationer / TensorBoardMonitor read the flow-aligned state, not the
# training objective
# ---------------------------------------------------------------------------


def test_validationer_preserves_clean_validation_loss_from_flow_state():
    class _TrainerWithFlowValidation:
        def __init__(self):
            self.stats = {}
            self.ep = 3
            self._last_flow_validation_state = {}

        def validation(self, fast=True):
            self._last_flow_validation_state = {
                "validation_loss": torch.tensor(1.25),
                "validation_onsite_loss": torch.tensor(0.5),
                "validation_hopping_loss": torch.tensor(2.0),
            }
            return torch.tensor(99.0)

    trainer = _TrainerWithFlowValidation()
    validationer = Validationer(interval=1)
    validationer.trainer = trainer

    value = validationer._get_value(field="iteration", time=10)
    assert _scalar(value) == pytest.approx(1.25)
    assert trainer.stats["validation_loss"]["last"] == pytest.approx(1.25)

    validationer.epoch(time=3)
    assert trainer.stats["validation_loss"]["epoch_mean"] == pytest.approx(1.25)


def test_non_metric_scheduler_does_not_reduce_compatible_scalar():
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([param], lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    trainer = object.__new__(MultiTrainer)
    trainer.update_lr_per_iter = True
    trainer.distributed_expert = True
    trainer.local_expert_idx = 0
    trainer.lr_schedulers = [scheduler]
    trainer.iter = 2
    trainer._tagger = _NoopTagger()
    trainer._mean_expert_dp_scalar = lambda value: pytest.fail(
        "metric-free scheduler must not all_reduce a scalar"
    )

    trainer._local_scheduler_step(None)


def test_disabled_iter_scheduler_does_not_request_metric():
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([param], lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    trainer = object.__new__(MultiTrainer)
    trainer.update_lr_per_iter = False
    trainer.distributed_expert = True
    trainer.local_expert_idx = 0
    trainer.lr_schedulers = [scheduler]

    assert trainer._local_scheduler_requires_metric() is False


def test_tensorboard_monitor_writes_fresh_validation_iter_tags():
    writes = []

    class _Writer:
        def add_scalar(self, tag, value, step):
            writes.append((tag, float(value), int(step)))

        def flush(self):
            pass

    monitor = object.__new__(TensorBoardMonitor)
    monitor.writer = _Writer()
    monitor.flush_every = 0
    monitor.trainer = SimpleNamespace(
        iter=1000, num_experts=0,
        stats={
            "validation_loss": {"last": 0.5, "last_updated": 1000},
            "validation_loss_opt": {"last": 0.7, "last_updated": 1000},
            "validation_onsite_loss": {"last": 0.2, "last_updated": 1000},
            "validation_hopping_loss": {"last": 0.3, "last_updated": 1000},
        },
    )

    monitor.iteration(time=1000)

    assert ("validation_loss_iter/iteration", 0.5, 1000) in writes
    assert ("validation_loss_opt_iter/iteration", 0.7, 1000) in writes
    assert ("validation_onsite_loss_iter/iteration", 0.2, 1000) in writes
    assert ("validation_hopping_loss_iter/iteration", 0.3, 1000) in writes


def test_tensorboard_register_writes_endpoint_metric_space_metadata():
    texts, scalars = [], []

    class _Writer:
        def add_text(self, tag, value, step):
            texts.append((tag, value, step))

        def add_scalar(self, tag, value, step):
            scalars.append((tag, value, step))

        def flush(self):
            pass

    monitor = object.__new__(TensorBoardMonitor)
    monitor.writer = _Writer()
    trainer = SimpleNamespace(endpoint_metric_spaces={"train": "block", "validation": "rme"})

    monitor.register(trainer)

    assert ("metadata/train_endpoint_metric_space", "block", 0) in texts
    assert ("metadata/validation_endpoint_metric_space", "rme", 0) in texts
    assert ("metadata/train_endpoint_metric_space_is_rme", 0.0, 0) in scalars
    assert ("metadata/validation_endpoint_metric_space_is_rme", 1.0, 0) in scalars


def test_tensorboard_monitor_writes_epoch_validation_on_iteration_axis():
    writes = []

    class _Writer:
        def add_scalar(self, tag, value, step):
            writes.append((tag, float(value), int(step)))

        def flush(self):
            pass

    monitor = object.__new__(TensorBoardMonitor)
    monitor.writer = _Writer()
    monitor.flush_every = 0
    monitor.trainer = SimpleNamespace(
        iter=4321, ep=4, num_experts=0,
        stats={
            "validation_loss": {"epoch_mean": 0.5, "last": 0.5, "epoch_last_updated": 4},
            "validation_onsite_loss": {"epoch_mean": 0.2, "last": 0.2, "epoch_last_updated": 4},
            "validation_hopping_loss": {"epoch_mean": 0.3, "last": 0.3, "epoch_last_updated": 4},
            "validation_compatible_euler_1_loss": {
                "epoch_mean": 0.5, "last": 0.5, "epoch_last_updated": 4
            },
            "validation_compatible_euler_1_onsite_loss": {
                "epoch_mean": 0.2, "last": 0.2, "epoch_last_updated": 4
            },
            "validation_compatible_euler_1_hopping_loss": {
                "epoch_mean": 0.3, "last": 0.3, "epoch_last_updated": 4
            },
            "validation_flow_one_step_loss": {
                "epoch_mean": 0.9, "last": 0.9, "epoch_last_updated": 4
            },
        },
    )

    monitor.epoch(time=4)

    assert ("validation_loss_iter/iteration", 0.5, 4321) in writes
    assert ("validation_onsite_loss_iter/iteration", 0.2, 4321) in writes
    assert ("validation_hopping_loss_iter/iteration", 0.3, 4321) in writes
    assert ("validation_compatible_euler_1_loss_iter/iteration", 0.5, 4321) not in writes
    assert ("validation_compatible_euler_1_onsite_loss_iter/iteration", 0.2, 4321) not in writes
    assert ("validation_compatible_euler_1_hopping_loss_iter/iteration", 0.3, 4321) not in writes
    assert ("validation_flow_one_step_loss_iter/iteration", 0.9, 4321) in writes


# ---------------------------------------------------------------------------
# Trainer._loss_on_batch: model-in-loss raw-batch fallback and its absence
# ---------------------------------------------------------------------------


def test_model_in_loss_skips_train_compatible_loss_from_raw_batch(monkeypatch):
    trainer = object.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.flow_cfm = _ModelInLossFlow()
    trainer.model = _UNUSED_MODEL

    def fake_to_dict(batch):
        return {"raw_batch": True}

    def fail_compatible(*args, **kwargs):
        raise AssertionError("model-in-loss pMF must not log raw-batch train compatible loss")

    monkeypatch.setattr(trainer_module.AtomicData, "to_AtomicDataDict", fake_to_dict)
    monkeypatch.setattr(Trainer, "_compatible_loss_state", staticmethod(fail_compatible))

    with pytest.raises(RuntimeError, match="could not reconstruct"):
        trainer._loss_on_batch(_FakeBatch(), _ComponentLoss())


def test_model_in_loss_train_loss_aligns_from_endpoint_stats(monkeypatch):
    trainer = object.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.flow_cfm = _ModelInLossFlowWithStats()
    trainer.model = _UNUSED_MODEL
    lossfunc = StatsCompatibleLoss()

    def fake_to_dict(batch):
        return {"raw_batch": True}

    def fail_compatible(*args, **kwargs):
        raise AssertionError("model-in-loss pMF must use flow stats, not raw-batch criterion")

    monkeypatch.setattr(trainer_module.AtomicData, "to_AtomicDataDict", fake_to_dict)
    monkeypatch.setattr(Trainer, "_compatible_loss_state", staticmethod(fail_compatible))

    loss = trainer._loss_on_batch(_FakeBatch(), lossfunc)
    state = trainer._last_flow_state

    onsite = 0.5 * (2.0 + (10.0 / 2.0) ** 0.5)
    hopping = 0.5 * (1.0 + 3.0 ** 0.5)
    aligned_total = 0.5 * (onsite + hopping)

    assert loss.item() == pytest.approx(7.0)
    assert lossfunc.forward_calls == 0
    assert lossfunc.stats_calls == 1
    assert state["train_loss_opt"].item() == pytest.approx(7.0)
    assert state["train_loss"].item() == pytest.approx(aligned_total)
    assert state["train_onsite_loss"].item() == pytest.approx(onsite)
    assert state["train_hopping_loss"].item() == pytest.approx(hopping)


class _ModelInLossFlowWithMismatchedStats(_ModelInLossFlow):
    """model_in_loss=True flow whose published stats declare the WRONG label.

    There is no raw-batch recompute fallback on the model_in_loss=True branch
    of Trainer._loss_on_batch, so this must fail fast with the specific
    metric-space-mismatch ValueError instead of the generic "could not
    reconstruct" RuntimeError three lines later.
    """

    def loss_with_model(self, model, batch, batch_for_loss):
        loss, state = super().loss_with_model(model, batch, batch_for_loss)
        stats = _compatible_clean_stats()
        stats["_compatible_clean_stats"]["metric_space"] = "rme"
        state.update(stats)
        return loss, state


def test_model_in_loss_train_loss_fails_fast_on_metric_space_mismatch(monkeypatch):
    trainer = object.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.flow_cfm = _ModelInLossFlowWithMismatchedStats()
    trainer.model = _UNUSED_MODEL
    lossfunc = BlockEndpointFallbackLoss()
    assert lossfunc.endpoint_metric_space == "block"

    def fake_to_dict(batch):
        return {"raw_batch": True}

    def fail_compatible(*args, **kwargs):
        raise AssertionError(
            "model_in_loss=True has no raw-batch fallback; a metric-space "
            "mismatch must raise before ever reaching it"
        )

    monkeypatch.setattr(trainer_module.AtomicData, "to_AtomicDataDict", fake_to_dict)
    monkeypatch.setattr(Trainer, "_compatible_loss_state", staticmethod(fail_compatible))

    with pytest.raises(ValueError, match=r"metric_space='rme'.*endpoint_metric_space='block'"):
        trainer._loss_on_batch(_FakeBatch(), lossfunc)


def test_loss_on_batch_can_skip_flow_for_reference_batch(monkeypatch):
    trainer = object.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.flow_cfm = _ModelInLossFlow()

    class ReferenceModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, batch):
            self.calls += 1
            pred = batch.copy()
            pred["node_features"] = pred["node_features"] + 2.0
            pred["edge_features"] = pred["edge_features"] + 3.0
            return pred

    def fail_loss_with_model(*args, **kwargs):
        raise AssertionError("reference batches should not enter pMF loss_with_model by default")

    def fake_to_dict(batch):
        return {
            "node_features": torch.tensor([[1.0]]),
            "edge_features": torch.tensor([[2.0]]),
        }

    model = ReferenceModel()
    trainer.model = model
    monkeypatch.setattr(trainer.flow_cfm, "loss_with_model", fail_loss_with_model)
    monkeypatch.setattr(trainer_module.AtomicData, "to_AtomicDataDict", fake_to_dict)

    loss = trainer._loss_on_batch(_FakeBatch(), _ComponentLoss(), use_flow=False)

    assert loss.item() == pytest.approx(2.5)
    assert trainer._last_flow_state == {}
    assert model.calls == 1


def test_iteration_reference_batch_does_not_overwrite_main_flow_state():
    trainer = object.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.model = torch.nn.Linear(1, 1, bias=False)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.flow_cfm = SimpleNamespace(enabled=True, apply_to_reference=False)
    trainer.train_lossfunc = object()
    trainer.reference_lossfunc = object()
    trainer.clip_grad_norm = 1.0
    trainer.update_lr_per_iter = False
    trainer.optimizer_diagnostics_freq = 999
    trainer.iter = 2
    trainer.num_experts = 0
    captured = {}

    def fake_loss_on_batch(self, batch, lossfunc, *, use_flow=True, allow_self_consistency=True):
        parameter = next(self.model.parameters())
        if use_flow:
            self._last_flow_state = {
                "train_loss": torch.tensor(2.0),
                "train_onsite_loss": torch.tensor(1.0),
                "train_hopping_loss": torch.tensor(3.0),
                "train_flow_loss": torch.tensor(7.0),
                "train_loss_opt": torch.tensor(7.0),
            }
            self._last_self_consistency_state = {}
            return parameter.sum() * 0.0 + 7.0
        self._last_flow_state = {}
        return parameter.sum() * 0.0 + 3.0

    trainer._loss_on_batch = fake_loss_on_batch.__get__(trainer, Trainer)
    trainer.call_plugins = lambda **kwargs: captured.update(kwargs)

    trainer.iteration(_FakeBatch(), _FakeBatch())

    assert captured["train_loss"].item() == pytest.approx(2.0)
    assert captured["train_onsite_loss"].item() == pytest.approx(1.0)
    assert captured["train_hopping_loss"].item() == pytest.approx(3.0)
    assert captured["train_flow_loss"].item() == pytest.approx(7.0)
    assert captured["train_loss_opt"].item() == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Pixel MeanFlow end-to-end through Trainer.validation()
# ---------------------------------------------------------------------------


class _ValidationConstantModel:
    """Constant x-prediction surrogate for end-to-end pMF validation."""

    def eval(self):
        return None

    def __call__(self, batch):
        out = batch.copy()
        out["node_features"] = torch.ones_like(batch["node_h0"])
        out["edge_features"] = torch.full_like(batch["edge_h0"], 2.0)
        return out


def _pixel_meanflow_validation_trainer(monkeypatch, flow_overrides=None):
    options = {
        "enabled": True, "objective": "pixel_meanflow", "mode": "residual", "prior": "zero",
        "strict_h0": True, "meanflow": {"fd_eps": 1.0e-4},
    }
    if flow_overrides:
        options["meanflow"].update(flow_overrides.pop("meanflow", {}))
        options.update(flow_overrides)

    trainer = object.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.dtype = torch.float32
    trainer.model = _ValidationConstantModel()
    trainer.flow_cfm = build_hamiltonian_flow(options)
    trainer.validation_loader = [_FakeBatch()]
    trainer.validation_lossfunc = _ComponentLoss()
    trainer.iter = 5

    monkeypatch.setattr(
        trainer_module.AtomicData, "to_AtomicDataDict", lambda batch: two_graph_batch()
    )
    return trainer


def test_pixel_meanflow_validation_writes_legacy_endpoint_compatible_keys(monkeypatch):
    trainer = _pixel_meanflow_validation_trainer(monkeypatch)

    loss = trainer.validation(fast=True)
    st = trainer._last_flow_validation_state

    # Legacy keys must carry the euler/endpoint blockwise compatible loss so pMF
    # curves line up with no-CFM/CFM validation semantics, and validation()
    # itself must return that aligned scalar rather than the flow objective.
    assert loss.item() == pytest.approx(1.5)
    assert st["validation_loss"].item() == pytest.approx(1.5)
    assert st["validation_onsite_loss"].item() == pytest.approx(1.0)
    assert st["validation_hopping_loss"].item() == pytest.approx(2.0)
    assert st["validation_compatible_euler_1_loss"].item() == pytest.approx(1.5)
    assert st["validation_compatible_euler_3_loss"].item() == pytest.approx(1.5)

    # The meanflow objective stays observable under validation_flow_* keys and
    # must not be what the legacy validation_loss reports.
    assert "validation_flow_random_t_loss" in st
    assert "validation_flow_one_step_loss" in st
    assert st["validation_flow_random_t_loss"].item() != pytest.approx(st["validation_loss"].item())
    assert "validation_flow_one_step_onsite_velocity_loss" in st
    assert not any(key.startswith("validation_one_step_flow_") for key in st)


class _ScalarOnlyLoss(torch.nn.Module):
    def forward(self, pred, ref):
        return (pred["node_features"] - ref["node_features"]).abs().mean()


def test_pixel_meanflow_validation_fails_closed_without_endpoint_components(monkeypatch):
    trainer = _pixel_meanflow_validation_trainer(monkeypatch)
    trainer.validation_lossfunc = _ScalarOnlyLoss()

    with pytest.raises(RuntimeError, match="endpoint triplet"):
        trainer.validation(fast=True)


def test_pixel_meanflow_validation_compatible_sampling_is_forced(monkeypatch):
    trainer = _pixel_meanflow_validation_trainer(
        monkeypatch, flow_overrides={"meanflow": {"log_validation_compatible_loss": False}}
    )
    sample_calls = []
    original_sample = trainer.flow_cfm.sample

    def counting_sample(model, batch, *, num_steps):
        sample_calls.append(int(num_steps))
        return original_sample(model, batch, num_steps=num_steps)

    monkeypatch.setattr(trainer.flow_cfm, "sample", counting_sample)

    trainer.validation(fast=True)
    st = trainer._last_flow_validation_state

    assert sample_calls == [1, 3]
    assert "validation_loss" in st
    assert "validation_onsite_loss" in st
    assert "validation_hopping_loss" in st
