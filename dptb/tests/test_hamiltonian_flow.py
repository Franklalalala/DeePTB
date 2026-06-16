from pathlib import Path
import importlib
from types import SimpleNamespace

import pytest
import torch

from dptb.nnops.flow import HamiltonianCFM, HamiltonianPixelMeanFlow, build_hamiltonian_flow
from dptb.nnops import trainer as trainer_module
from dptb.nnops.trainer import Trainer
from dptb.utils.argcheck import flow_options

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
    assert "train_onsite_loss" not in state
    assert "train_hopping_loss" not in state


def test_cfm_keeps_flow_metrics_out_of_legacy_train_tags_and_router_stats():
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
    assert "train_onsite_loss" not in state
    assert "train_hopping_loss" not in state
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


def test_flow_compatible_loss_state_can_write_legacy_only_tags():
    lossfunc = _ComponentLoss()
    pred, ref = _pred_ref()

    state = Trainer._compatible_loss_state(
        lossfunc,
        pred,
        ref,
        prefix="train",
        legacy_prefix=None,
    )

    assert state["train_loss"].item() == pytest.approx(2.0)
    assert state["train_onsite_loss"].item() == pytest.approx(2.0)
    assert state["train_hopping_loss"].item() == pytest.approx(2.0)
    assert not any(key.startswith("train_compatible") for key in state)


def test_flow_validation_compatible_state_maps_euler1_to_canonical_tags_when_logging():
    trainer = object.__new__(Trainer)
    trainer.flow_cfm = SimpleNamespace(
        compatible_loss_to_legacy_keys=False,
        log_validation_compatible_loss=True,
    )
    lossfunc = _ComponentLoss()
    pred, ref = _pred_ref()

    state = trainer._flow_validation_compatible_state(
        lossfunc,
        pred,
        ref,
        num_steps=1,
    )

    assert state["validation_loss"].item() == pytest.approx(2.0)
    assert state["validation_onsite_loss"].item() == pytest.approx(2.0)
    assert state["validation_hopping_loss"].item() == pytest.approx(2.0)
    assert not any(key.startswith("validation_compatible") for key in state)


def test_flow_validation_compatible_state_prefers_legacy_tags_without_extra():
    trainer = object.__new__(Trainer)
    trainer.flow_cfm = SimpleNamespace(
        compatible_loss_to_legacy_keys=True,
        log_validation_compatible_loss=True,
    )
    lossfunc = _ComponentLoss()
    pred, ref = _pred_ref()

    state = trainer._flow_validation_compatible_state(
        lossfunc,
        pred,
        ref,
        num_steps=1,
    )

    assert state["validation_loss"].item() == pytest.approx(2.0)
    assert state["validation_onsite_loss"].item() == pytest.approx(2.0)
    assert state["validation_hopping_loss"].item() == pytest.approx(2.0)
    assert not any(key.startswith("validation_compatible") for key in state)


def test_flow_compatible_loss_state_maps_validation_components_only():
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
    assert "validation_loss" not in state


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


def test_flow_options_argcheck_accepts_pixel_meanflow_config_keys():
    schema = flow_options()

    value = schema.normalize_value(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "flow_time_r_key": "flow_time_r",
            "flow_time_t_key": "flow_time_t",
            "flow_time_h_key": "flow_time_h",
            "meanflow": {
                "profile": "conservative",
                "jvp_tangent": "boundary",
                "du_dt_backend": "finite_difference",
            },
        }
    )
    schema.check_value(value, strict=True)

    assert value["meanflow"]["profile"] == "conservative"


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


class _EndpointOffsetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))
        self.calls = 0

    def forward(self, data):
        self.calls += 1
        pred = data.copy()
        pred["node_features"] = torch.full_like(data["node_features"], 2.0) + self.bias
        pred["edge_features"] = torch.full_like(data["edge_features"], 4.0) + self.bias
        return pred


def test_loss_on_batch_logs_legacy_compatible_loss_without_extra_tags(monkeypatch):
    trainer = object.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.model = _EndpointOffsetModel()
    trainer.flow_cfm = HamiltonianCFM(
        {
            "enabled": True,
            "mode": "full",
            "prior": "zero",
            "omit_time_scaling": True,
            "loss_type": "mse",
            "log_train_compatible_loss": False,
            "compatible_loss_to_legacy_keys": True,
        }
    )
    trainer.flow_cfm._sample_t = lambda *, num_graphs, device, dtype: torch.zeros(
        num_graphs, device=device, dtype=dtype
    )

    def fake_to_dict(batch):
        return {
            "batch": torch.zeros(2, dtype=torch.long),
            "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
            "node_h0": torch.zeros(2, 1),
            "edge_h0": torch.zeros(1, 1),
            "node_features": torch.zeros(2, 1),
            "edge_features": torch.zeros(1, 1),
        }

    monkeypatch.setattr(trainer_module.AtomicData, "to_AtomicDataDict", fake_to_dict)

    loss = trainer._loss_on_batch(_FakeBatch(), _ComponentLoss())
    state = trainer._last_flow_state

    assert trainer.model.calls == 1
    assert loss.item() == pytest.approx(8.0)
    assert state["train_flow_loss"].item() == pytest.approx(8.0)
    assert state["train_flow_onsite_loss"].item() == pytest.approx(4.0)
    assert state["train_flow_hopping_loss"].item() == pytest.approx(16.0)
    assert state["train_loss"].item() == pytest.approx(3.0)
    assert state["train_onsite_loss"].item() == pytest.approx(2.0)
    assert state["train_hopping_loss"].item() == pytest.approx(4.0)
    assert not any(key.startswith("train_compatible") for key in state)


def test_loss_on_batch_maps_flow_components_to_canonical_train_tags_when_compatible_disabled(monkeypatch):
    trainer = object.__new__(Trainer)
    trainer.device = torch.device("cpu")
    trainer.model = _EndpointOffsetModel()
    trainer.flow_cfm = HamiltonianCFM(
        {
            "enabled": True,
            "mode": "full",
            "prior": "zero",
            "omit_time_scaling": True,
            "loss_type": "mse",
            "log_train_compatible_loss": False,
            "compatible_loss_to_legacy_keys": False,
        }
    )
    trainer.flow_cfm._sample_t = lambda *, num_graphs, device, dtype: torch.zeros(
        num_graphs, device=device, dtype=dtype
    )

    def fake_to_dict(batch):
        return {
            "batch": torch.zeros(2, dtype=torch.long),
            "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
            "node_h0": torch.zeros(2, 1),
            "edge_h0": torch.zeros(1, 1),
            "node_features": torch.zeros(2, 1),
            "edge_features": torch.zeros(1, 1),
        }

    monkeypatch.setattr(trainer_module.AtomicData, "to_AtomicDataDict", fake_to_dict)

    loss = trainer._loss_on_batch(_FakeBatch(), _ComponentLoss())
    state = trainer._last_flow_state

    assert loss.item() == pytest.approx(8.0)
    assert state["train_flow_loss"].item() == pytest.approx(8.0)
    assert state["train_flow_onsite_loss"].item() == pytest.approx(4.0)
    assert state["train_flow_hopping_loss"].item() == pytest.approx(16.0)
    assert state["train_loss"].item() == pytest.approx(8.0)
    assert state["train_onsite_loss"].item() == pytest.approx(4.0)
    assert state["train_hopping_loss"].item() == pytest.approx(16.0)
    assert not any(key.startswith("train_compatible") for key in state)


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


def test_pixel_meanflow_endpoint_counts_match_cfm_compatible_validation_semantics():
    flow = HamiltonianPixelMeanFlow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "mode": "residual",
            "prior": "zero",
            "strict_h0": True,
        }
    )
    clean = torch.zeros(4, 1)
    prior = torch.zeros_like(clean)
    state_z = torch.zeros_like(clean)
    pred_x = torch.tensor([[1.0], [3.0], [100.0], [100.0]])
    mask = torch.tensor([[True], [True], [False], [False]])

    _, state = flow._component_meanflow_loss(
        diff_prefix="validation_flow_hopping",
        pred_x=pred_x,
        boundary_x=None,
        clean=clean,
        prior=prior,
        state_z=state_z,
        comp_r=torch.full((4,), 0.25),
        comp_t=torch.full((4,), 0.50),
        pred_x_eps=pred_x,
        mask=mask,
        weight=1.0,
    )

    assert state["validation_flow_hopping_endpoint_loss"].item() == pytest.approx(5.0)
    assert state["validation_flow_hopping_endpoint_mse"].item() == pytest.approx(5.0)
    assert state["validation_flow_hopping_endpoint_mae"].item() == pytest.approx(2.0)
    assert state["validation_flow_hopping_endpoint_l1_sum"].item() == pytest.approx(4.0)
    assert state["validation_flow_hopping_endpoint_mse_sum"].item() == pytest.approx(10.0)
    assert state["validation_flow_hopping_endpoint_count"].item() == pytest.approx(2.0)


def test_pixel_meanflow_train_legacy_tags_use_endpoint_residual_loss():
    flow = HamiltonianPixelMeanFlow(
        {
            "enabled": True,
            "objective": "pixel_meanflow",
            "mode": "residual",
            "prior": "zero",
            "strict_h0": True,
            "meanflow": {"aux_endpoint_weight": 0.0},
        }
    )
    model = _EndpointOffsetModel()
    model.bias.data.fill_(1.0)

    _, state = flow.loss_with_model(
        model,
        _two_graph_batch(),
        _two_graph_ref(),
        r=torch.tensor([0.25, 0.25]),
        t=torch.tensor([0.50, 0.50]),
    )

    assert state["train_flow_onsite_endpoint_loss"].item() == pytest.approx(1.0)
    assert state["train_flow_hopping_endpoint_loss"].item() == pytest.approx(1.0)
    assert state["train_onsite_loss"].item() == pytest.approx(1.0)
    assert state["train_hopping_loss"].item() == pytest.approx(1.0)


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
