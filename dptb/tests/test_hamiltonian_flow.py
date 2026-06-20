from pathlib import Path
import importlib
from types import SimpleNamespace

import pytest
import torch

from dptb.nnops.flow import HamiltonianCFM, HamiltonianPixelMeanFlow, build_hamiltonian_flow
from dptb.data import AtomicDataDict
from dptb.nnops.loss import HamilLossAbs
from dptb.nnops import trainer as trainer_module
from dptb.nnops.multi_trainer import MultiTrainer
from dptb.nnops.trainer import Trainer
from dptb.plugins.monitor import TensorBoardMonitor

train_entrypoint = importlib.import_module("dptb.entrypoints.train")


def _two_graph_batch():
    return {
        "batch": torch.tensor([0, 0, 1], dtype=torch.long),
        "edge_index": torch.tensor([[0, 2], [1, 2]], dtype=torch.long),
        "node_h0": torch.zeros(3, 1),
        "edge_h0": torch.zeros(2, 1),
        "node_features": torch.zeros(3, 1),
        "edge_features": torch.zeros(2, 1),
    }


def _two_graph_ref():
    return {
        "batch": torch.tensor([0, 0, 1], dtype=torch.long),
        "edge_index": torch.tensor([[0, 2], [1, 2]], dtype=torch.long),
        "node_features": torch.full((3, 1), 2.0),
        "edge_features": torch.full((2, 1), 4.0),
    }


def test_prepare_batch_samples_and_expands_time_per_graph():
    flow = HamiltonianCFM(
        {
            "enabled": True,
            "prior": "zero",
            "omit_time_scaling": True,
            "strict_h0": True,
        }
    )
    flow._sample_t = lambda *, num_graphs, device, dtype: torch.tensor(
        [0.0, 0.5], device=device, dtype=dtype
    )

    data, ref, ctx = flow.prepare_batch(_two_graph_batch(), _two_graph_ref())

    assert ctx.t.shape == (2,)
    assert torch.equal(ctx.node_t, torch.tensor([0.0, 0.0, 0.5]))
    assert torch.equal(ctx.edge_t, torch.tensor([0.0, 0.5]))
    assert torch.equal(data["flow_time"], torch.tensor([0.0, 0.5]))
    assert torch.equal(ref["flow_time"], torch.tensor([0.0, 0.5]))
    assert torch.equal(data["node_h0"].flatten(), torch.tensor([0.0, 0.0, 1.0]))
    assert torch.equal(data["edge_h0"].flatten(), torch.tensor([0.0, 2.0]))


def test_residual_flow_fails_fast_when_h0_is_missing():
    flow = HamiltonianCFM({"enabled": True, "mode": "residual", "strict_h0": True})
    data = _two_graph_batch()
    data.pop("node_h0")

    with pytest.raises(KeyError, match="node_h0"):
        flow.prepare_batch(data, _two_graph_ref())


def test_global_element_reduction_does_not_equal_weight_node_and_edge_components():
    flow = HamiltonianCFM(
        {
            "enabled": True,
            "omit_time_scaling": True,
            "component_reduction": "global_elements",
        }
    )
    data = {
        "batch": torch.tensor([0], dtype=torch.long),
        "edge_index": torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.long),
        "node_h0": torch.zeros(1, 1),
        "edge_h0": torch.zeros(3, 1),
        "node_features": torch.zeros(1, 1),
        "edge_features": torch.zeros(3, 1),
    }
    ref = {
        "batch": data["batch"],
        "edge_index": data["edge_index"],
        "node_features": torch.zeros(1, 1),
        "edge_features": torch.zeros(3, 1),
    }
    _, ref, ctx = flow.prepare_batch(data, ref, t=torch.zeros(1))
    pred = {
        "node_features": torch.ones(1, 1),
        "edge_features": torch.full((3, 1), 3.0),
    }

    loss, state = flow.loss(pred, ref, ctx)

    assert loss.item() == pytest.approx(7.0)
    assert state["train_flow_onsite_loss"].item() == pytest.approx(1.0)
    assert state["train_flow_hopping_loss"].item() == pytest.approx(9.0)
    assert state["train_onsite_loss"].item() == pytest.approx(1.0)
    assert state["train_hopping_loss"].item() == pytest.approx(9.0)


def test_cfm_writes_default_legacy_train_tags_and_router_stats():
    flow = HamiltonianCFM(
        {
            "enabled": True,
            "omit_time_scaling": True,
        }
    )
    data, ref, ctx = flow.prepare_batch(_two_graph_batch(), _two_graph_ref(), t=torch.zeros(2))
    pred = {
        "batch": data["batch"],
        "edge_index": data["edge_index"],
        "node_features": ref["node_features"] + 1.0,
        "edge_features": ref["edge_features"] + 3.0,
        "mean_max_prob": torch.tensor(0.75),
        "expert_load_cv": torch.tensor(0.25),
    }

    _, state = flow.loss(pred, ref, ctx)

    assert state["train_flow_onsite_loss"].item() == pytest.approx(1.0)
    assert state["train_flow_hopping_loss"].item() == pytest.approx(9.0)
    assert state["train_onsite_loss"].item() == pytest.approx(1.0)
    assert state["train_hopping_loss"].item() == pytest.approx(9.0)
    assert state["mean_max_prob"].item() == pytest.approx(0.75)
    assert state["expert_load_cv"].item() == pytest.approx(0.25)


def test_single_trainer_effective_expert_lr_state_uses_global_optimizer_lr():
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([param], lr=0.0125)
    state = {}

    Trainer._add_effective_expert_lr_state(state, optimizer=optimizer, num_experts=2)

    assert state["expert_0_lr"] == pytest.approx(0.0125)
    assert state["expert_1_lr"] == pytest.approx(0.0125)


def test_single_train_entrypoint_passes_train_options_to_build_model():
    text = Path(train_entrypoint.__file__).read_text(encoding="utf-8")
    build_call = text[text.index("model = build_model("): text.index("trainer = Trainer(")]

    assert 'train_options=jdata["train_options"]' in build_call


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


class _StatsCompatibleLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_calls = 0
        self.stats_calls = 0
        self.onsite_boost = False
        self.element_average = False
        self.z_loss_coef = 0.0

    def forward(self, pred, ref):
        self.forward_calls += 1
        raise AssertionError("compatible logging must not re-run the full criterion")

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
        self.stats_calls += 1
        onsite = 0.5 * (
            onsite_l1_sum / onsite_count.clamp_min(1.0)
            + torch.sqrt(onsite_mse_sum / onsite_count.clamp_min(1.0) + 1e-12)
        )
        hopping = 0.5 * (
            hopping_l1_sum / hopping_count.clamp_min(1.0)
            + torch.sqrt(hopping_mse_sum / hopping_count.clamp_min(1.0) + 1e-12)
        )
        return 0.5 * (onsite + hopping), onsite, hopping


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


class _LossIDP:
    def __init__(self):
        self.mask_to_nrme = torch.tensor(
            [[True, True, False], [True, False, True]]
        )
        self.mask_to_erme = torch.tensor(
            [[True, False, True], [False, True, True]]
        )


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
    node_mask = idp.mask_to_nrme[pred[AtomicDataDict.ATOM_TYPE_KEY].flatten()]
    edge_mask = idp.mask_to_erme[pred[AtomicDataDict.EDGE_TYPE_KEY].flatten()]
    node_stats = _masked_stats(
        pred[AtomicDataDict.NODE_FEATURES_KEY] - ref[AtomicDataDict.NODE_FEATURES_KEY],
        node_mask,
    )
    edge_stats = _masked_stats(
        pred[AtomicDataDict.EDGE_FEATURES_KEY] - ref[AtomicDataDict.EDGE_FEATURES_KEY],
        edge_mask,
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
    torch.testing.assert_close(stats_onsite, lossfunc.last_onsite_loss)
    torch.testing.assert_close(stats_hopping, lossfunc.last_hopping_loss)


def _pred_ref():
    pred = {
        "node_features": torch.tensor([[1.0], [3.0]], requires_grad=True),
        "edge_features": torch.tensor([[2.0]], requires_grad=True),
    }
    ref = {
        "node_features": torch.zeros(2, 1),
        "edge_features": torch.zeros(1, 1),
    }
    return pred, ref


def test_flow_compatible_loss_state_uses_no_grad_and_restores_side_effects():
    lossfunc = _ComponentLoss()
    pred, ref = _pred_ref()

    state = Trainer._compatible_loss_state(
        lossfunc,
        pred,
        ref,
        prefix="train_compatible",
        legacy_prefix=None,
    )

    assert lossfunc.called_with_grad_enabled is False
    assert state["train_compatible_loss"].requires_grad is False
    assert state["train_compatible_onsite_loss"].item() == pytest.approx(2.0)
    assert state["train_compatible_hopping_loss"].item() == pytest.approx(2.0)
    assert "train_onsite_loss" not in state
    assert "train_hopping_loss" not in state

    assert lossfunc.last_onsite_loss.item() == pytest.approx(123.0)
    assert lossfunc.last_hopping_loss.item() == pytest.approx(456.0)


def test_flow_stats_fast_path_preserves_compatible_and_legacy_semantics():
    lossfunc = _StatsCompatibleLoss()
    state = Trainer._compatible_loss_state_from_flow_stats(
        lossfunc,
        _compatible_clean_stats(),
        source_prefix="train",
        prefix="train_compatible",
        legacy_prefix="train",
        global_step=17,
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


def test_flow_compatible_loss_state_explicit_legacy_mapping():
    lossfunc = _ComponentLoss()
    pred, ref = _pred_ref()

    state = Trainer._compatible_loss_state(
        lossfunc,
        pred,
        ref,
        prefix="train_compatible",
        legacy_prefix="train",
    )

    assert state["train_onsite_loss"].item() == pytest.approx(2.0)
    assert state["train_hopping_loss"].item() == pytest.approx(2.0)
    assert lossfunc.last_onsite_loss.item() == pytest.approx(123.0)
    assert lossfunc.last_hopping_loss.item() == pytest.approx(456.0)


def test_flow_compatible_loss_state_maps_validation_clean_legacy_loss():
    lossfunc = _ComponentLoss()
    pred, ref = _pred_ref()

    state = Trainer._compatible_loss_state(
        lossfunc,
        pred,
        ref,
        prefix="validation_compatible_euler_1",
        legacy_prefix="validation",
    )

    assert state["validation_onsite_loss"].item() == pytest.approx(2.0)
    assert state["validation_hopping_loss"].item() == pytest.approx(2.0)
    assert state["validation_loss"].item() == pytest.approx(2.0)


class _ConstantEndpoint(torch.nn.Module):
    def forward(self, data):
        data = data.copy()
        data["node_features"] = torch.full_like(data["node_h0"], 2.0)
        data["edge_features"] = torch.full_like(data["edge_h0"], 4.0)
        return data


@pytest.mark.parametrize("num_steps", [1, 3])
def test_euler_sampler_reaches_constant_predicted_endpoint(num_steps):
    flow = HamiltonianCFM(
        {
            "enabled": True,
            "prior": "zero",
            "omit_time_scaling": True,
            "strict_h0": True,
        }
    )

    sampled = flow.sample(_ConstantEndpoint(), _two_graph_batch(), num_steps=num_steps)

    assert torch.allclose(sampled["node_features"], torch.full((3, 1), 2.0))
    assert torch.allclose(sampled["edge_features"], torch.full((2, 1), 4.0))
    assert torch.equal(sampled["flow_time"], torch.ones(2))


def test_build_hamiltonian_flow_selects_pixel_meanflow_objective():
    flow = build_hamiltonian_flow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "mode": "residual",
            "prior": "zero",
        }
    )

    assert isinstance(flow, HamiltonianPixelMeanFlow)
    assert flow.model_in_loss is True


def test_pixel_meanflow_conservative_defaults_to_paper_boundary_tangent():
    flow = HamiltonianPixelMeanFlow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "meanflow": {"profile": "conservative"},
        }
    )

    assert flow.meanflow_profile == "conservative"
    assert flow.meanflow_jvp_tangent == "boundary"
    assert flow.meanflow_norm_p == pytest.approx(0.0)
    assert flow.meanflow_aux_boundary_v_weight == pytest.approx(0.0)


def test_pixel_meanflow_du_dt_backend_is_explicit_finite_difference():
    flow = HamiltonianPixelMeanFlow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "meanflow": {"du_dt_backend": "finite_difference"},
        }
    )

    assert flow.meanflow_du_dt_backend == "finite_difference"

    with pytest.raises(NotImplementedError, match="finite_difference"):
        HamiltonianPixelMeanFlow(
            {
                "enabled": True,
                "objective": "pixel_meanflow",
                "meanflow": {"du_dt_backend": "jvp"},
            }
        )


def test_pixel_meanflow_aggressive_profile_sets_opt_in_knobs():
    flow = HamiltonianPixelMeanFlow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "meanflow": {"profile": "aggressive"},
        }
    )

    assert flow.meanflow_profile == "aggressive"
    assert flow.meanflow_jvp_tangent == "boundary"
    assert flow.meanflow_norm_p == pytest.approx(1.0)
    assert flow.meanflow_aux_boundary_v_weight > 0.0


def test_flow_apply_to_reference_defaults_false_and_can_opt_in():
    default_flow = HamiltonianPixelMeanFlow({"enabled": True, "objective": "pixel_meanflow"})
    opt_in_flow = HamiltonianPixelMeanFlow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "apply_to_reference": True,
        }
    )

    assert default_flow.apply_to_reference is False
    assert opt_in_flow.apply_to_reference is True


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


class _FakeBatch:
    __slices__ = {}
    __cumsum__ = {}
    __cat_dims__ = {}
    __num_nodes_list__ = []
    __data_class__ = object

    def to(self, device):
        return self


_UNUSED_MODEL = object()


class _NoopTagger:
    def tag(self, *args, **kwargs):
        class _Ctx:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, tb):
                return False

        return _Ctx()


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

    batch = _two_graph_batch()
    result = trainer._run_one_expert_loss(
        batch,
        batch_info={},
        criterion=_ComponentLoss(),
        expert_idx=0,
        range_dis=(0.0, 1.0),
        capture_metrics=True,
    )

    assert trainer.flow_cfm.prepare_called
    assert trainer.flow_cfm.loss_called
    assert torch.equal(trainer.model.seen["node_h0"], batch["node_h0"] + 10.0)
    assert torch.equal(trainer.model.seen["edge_h0"], batch["edge_h0"] + 20.0)
    assert result["loss"].item() == pytest.approx(5.0)
    assert result["onsite"].item() == pytest.approx(2.0)
    assert result["hopping"].item() == pytest.approx(3.0)


def test_multitrainer_non_display_step_does_not_run_full_compatible_loss():
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
    lossfunc = _StatsCompatibleLoss()

    result = trainer._run_one_expert_loss(
        _two_graph_batch(),
        batch_info={},
        criterion=lossfunc,
        expert_idx=0,
        range_dis=(0.0, 1.0),
        capture_metrics=True,
    )

    assert lossfunc.forward_calls == 0
    assert lossfunc.stats_calls == 1
    assert result["last_onsite_count"].item() == pytest.approx(2.0)
    assert result["last_hopping_count"].item() == pytest.approx(3.0)
    assert result["onsite"].item() > 0.0
    assert result["hopping"].item() > 0.0


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
        iter=1000,
        num_experts=0,
        stats={
            "validation_loss": {"last": 0.5, "last_updated": 1000},
            "validation_onsite_loss": {"last": 0.2, "last_updated": 1000},
            "validation_hopping_loss": {"last": 0.3, "last_updated": 1000},
        },
    )

    monitor.iteration(time=1000)

    assert ("validation_loss_iter/iteration", 0.5, 1000) in writes
    assert ("validation_onsite_loss_iter/iteration", 0.2, 1000) in writes
    assert ("validation_hopping_loss_iter/iteration", 0.3, 1000) in writes


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

    loss = trainer._loss_on_batch(_FakeBatch(), _ComponentLoss())

    assert loss.item() == pytest.approx(7.0)
    assert trainer._last_flow_state["train_flow_loss"].item() == pytest.approx(7.0)


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


def test_pixel_meanflow_oracle_endpoint_has_zero_velocity_loss():
    flow = HamiltonianPixelMeanFlow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "mode": "residual",
            "prior": "zero",
            "strict_h0": True,
            "meanflow": {
                "aux_endpoint_weight": 0.0,
                "jvp_backend": "finite_difference",
                "fd_eps": 1.0e-4,
            },
        }
    )
    r = torch.tensor([0.2, 0.3])
    t = torch.tensor([0.5, 0.7])

    loss, state = flow.loss_with_model(_ConstantEndpoint(), _two_graph_batch(), _two_graph_ref(), r=r, t=t)

    assert loss.item() == pytest.approx(0.0, abs=1.0e-6)
    assert state["train_flow_h"].item() == pytest.approx(float((t - r).mean()), abs=1.0e-6)
    assert state["train_flow_onsite_velocity_mse"].item() == pytest.approx(0.0, abs=1.0e-6)
    assert state["train_flow_hopping_velocity_mse"].item() == pytest.approx(0.0, abs=1.0e-6)


def test_pixel_meanflow_one_step_sampler_reaches_constant_endpoint():
    flow = HamiltonianPixelMeanFlow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "mode": "residual",
            "prior": "zero",
            "strict_h0": True,
        }
    )

    sampled = flow.sample(_ConstantEndpoint(), _two_graph_batch(), num_steps=1)

    assert torch.allclose(sampled["node_features"], torch.full((3, 1), 2.0))
    assert torch.allclose(sampled["edge_features"], torch.full((2, 1), 4.0))
    assert torch.equal(sampled["flow_time"], torch.zeros(2))
    assert torch.equal(sampled["flow_time_r"], torch.zeros(2))
    assert torch.equal(sampled["flow_time_h"], torch.zeros(2))
