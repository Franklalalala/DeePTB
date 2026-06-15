import pytest
import torch

from dptb.data import _keys
from dptb.nnops.flow import build_hamiltonian_flow
from dptb.nnops.rmf import EuclideanManifold, HamiltonianRiemannianMeanFlow


def _flow_options(**extra):
    options = {
        "enabled": True,
        "type": "rmf",
        "objective": "rmf",
        "mode": "residual",
        "prior": "zero",
        "manifold": "euclidean",
        "loss_type": "mse",
        "detach_interpolated_h0": False,
        "meanflow": {
            "fd_eps": 1.0e-3,
            "aux_endpoint_weight": 0.1,
            "aux_boundary_v_weight": 0.0,
            "jvp_tangent": "path",
            "norm_p": 0.0,
        },
        "time_sampler": {
            "type": "uniform",
            "min_t": 0.05,
            "same_time_probability": 0.5,
        },
    }
    options.update(extra)
    return options


def _toy_batch():
    node_h0 = torch.tensor([[10.0, 11.0], [12.0, 13.0]])
    edge_h0 = torch.tensor([[20.0, 21.0], [22.0, 23.0]])
    node_res = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    edge_res = torch.tensor([[0.5, -0.5], [1.5, -1.5]])
    data = {
        _keys.BATCH_KEY: torch.zeros(2, dtype=torch.long),
        _keys.EDGE_INDEX_KEY: torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        _keys.NODE_H0_KEY: node_h0,
        _keys.EDGE_H0_KEY: edge_h0,
        _keys.NODE_FEATURES_KEY: node_h0.clone(),
        _keys.EDGE_FEATURES_KEY: edge_h0.clone(),
    }
    ref = {
        _keys.BATCH_KEY: data[_keys.BATCH_KEY],
        _keys.EDGE_INDEX_KEY: data[_keys.EDGE_INDEX_KEY],
        _keys.NODE_FEATURES_KEY: node_h0 + node_res,
        _keys.EDGE_FEATURES_KEY: edge_h0 + edge_res,
    }
    return data, ref, node_res, edge_res


class _EndpointModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.node_shift = torch.nn.Parameter(torch.tensor([[0.1, -0.2]]))
        self.edge_shift = torch.nn.Parameter(torch.tensor([[0.3, -0.4]]))

    def forward(self, data):
        out = data.copy()
        out[_keys.NODE_FEATURES_KEY] = data[_keys.NODE_H0_KEY] + self.node_shift
        out[_keys.EDGE_FEATURES_KEY] = data[_keys.EDGE_H0_KEY] + self.edge_shift
        out["mean_max_prob"] = self.node_shift.square().mean()
        out["expert_load_cv"] = self.edge_shift.square().mean()
        return out


def test_euclidean_manifold_ops_preserve_shapes_and_values():
    manifold = EuclideanManifold()
    x0 = torch.randn(4, 3, 2)
    x1 = torch.randn(4, 3, 2)
    t = torch.linspace(0.0, 1.0, 4)

    z = manifold.geodesic_interpolate(x0, x1, t)
    v = manifold.tangent_velocity(x0, x1, t)
    y = manifold.expmap(x0, manifold.logmap(x0, x1))

    assert z.shape == x0.shape
    assert v.shape == x0.shape
    assert y.shape == x0.shape
    assert torch.allclose(y, x1)
    assert torch.allclose(z[0], x0[0])
    assert torch.allclose(z[-1], x1[-1])


def test_build_hamiltonian_flow_accepts_rmf_type_alias():
    flow = build_hamiltonian_flow(_flow_options())
    assert isinstance(flow, HamiltonianRiemannianMeanFlow)


def test_rmf_prepare_batch_uses_h0_residual_geodesic_path_and_velocity():
    data, ref, node_res, edge_res = _toy_batch()
    flow = HamiltonianRiemannianMeanFlow(_flow_options())
    t = torch.tensor([0.25])
    prepared, _, ctx = flow.prepare_batch(data, ref, r=torch.tensor([0.0]), t=t)

    expected_node_state = (1.0 - t.item()) * node_res
    expected_edge_state = (1.0 - t.item()) * edge_res

    assert torch.allclose(ctx.node_state, expected_node_state)
    assert torch.allclose(ctx.edge_state, expected_edge_state)
    assert torch.allclose(prepared[_keys.NODE_H0_KEY], data[_keys.NODE_H0_KEY] + expected_node_state)
    assert torch.allclose(prepared[_keys.EDGE_H0_KEY], data[_keys.EDGE_H0_KEY] + expected_edge_state)
    assert torch.allclose(ctx.node_target_velocity, -node_res)
    assert torch.allclose(ctx.edge_target_velocity, -edge_res)


def test_rmf_forward_loss_backward_and_legacy_logging_state():
    data, ref, _, _ = _toy_batch()
    model = _EndpointModel()
    flow = HamiltonianRiemannianMeanFlow(_flow_options())

    loss, state = flow.loss_with_model(
        model,
        data,
        ref,
        r=torch.tensor([0.1]),
        t=torch.tensor([0.4]),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert model.node_shift.grad is not None
    assert model.edge_shift.grad is not None
    assert "train_flow_onsite_velocity_loss" in state
    assert "train_flow_hopping_velocity_loss" in state
    assert "train_onsite_loss" in state
    assert "train_hopping_loss" in state
    assert "mean_max_prob" in state
    assert "expert_load_cv" in state
