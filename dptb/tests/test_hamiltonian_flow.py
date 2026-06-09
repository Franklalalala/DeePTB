import pytest
import torch

from dptb.nnops.flow import HamiltonianCFM
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
