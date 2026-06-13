import pytest
import torch

from dptb.nnops.flow import HamiltonianCFM, HamiltonianPixelMeanFlow, build_hamiltonian_flow
from dptb.nnops import trainer as trainer_module
from dptb.nnops.trainer import Trainer


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


class _ModelInLossFlow:
    enabled = True
    model_in_loss = True
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
