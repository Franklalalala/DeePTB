import pytest
import torch

from dptb.nnops.flow_diagnostics import (
    cfm_chord_cosine_diagnostics,
    cosine_similarity_tensors,
    grad_cosine,
    pixel_meanflow_du_dt_diagnostics,
)


def test_cosine_similarity_tensors_flattens_complex_as_real_components():
    a = torch.tensor([1.0 + 2.0j, 3.0 + 4.0j])
    b = torch.tensor([1.0 + 2.0j, 3.0 + 4.0j])

    assert cosine_similarity_tensors(a, b).item() == pytest.approx(1.0)


def test_pixel_meanflow_du_dt_diagnostics_reports_norm_ratio_and_grad_cosine():
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    flow_loss = (param.square()).sum()
    jvp_loss = -(param.square()).sum()

    state = pixel_meanflow_du_dt_diagnostics(
        target_v=torch.tensor([3.0, 4.0]),
        du_dt=torch.tensor([0.0, 5.0]),
        flow_loss=flow_loss,
        jvp_loss=jvp_loss,
        parameters=[param],
    )

    assert state["du_dt_norm"].item() == pytest.approx(5.0)
    assert state["target_v_norm"].item() == pytest.approx(5.0)
    assert state["du_dt_norm_over_target_v_norm"].item() == pytest.approx(1.0)
    assert state["grad_cos_flow_jvp"].item() == pytest.approx(-1.0)


def test_grad_cosine_returns_nan_for_missing_gradients():
    param = torch.nn.Parameter(torch.tensor([1.0]))
    first = torch.tensor(1.0, requires_grad=True)
    second = torch.tensor(2.0, requires_grad=True)

    assert torch.isnan(grad_cosine(first, second, [param]))


def test_cfm_chord_cosine_diagnostics_is_one_for_exact_endpoint_direction():
    current = torch.zeros(2, 2)
    target = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    endpoint = target.clone()
    t = torch.tensor([0.25, 0.5])

    state = cfm_chord_cosine_diagnostics(
        node_current=current,
        node_target=target,
        node_endpoint=endpoint,
        t=t,
    )

    assert state["node_cos_v_theta_chord"].item() == pytest.approx(1.0)
    assert state["cos_v_theta_chord"].item() == pytest.approx(1.0)


def test_cfm_chord_cosine_diagnostics_aggregates_node_and_edge_components():
    node_current = torch.zeros(1, 2)
    node_target = torch.tensor([[1.0, 0.0]])
    node_endpoint = torch.tensor([[0.0, 1.0]])
    edge_current = torch.zeros(1, 2)
    edge_target = torch.tensor([[0.0, 2.0]])
    edge_endpoint = torch.tensor([[0.0, 4.0]])

    state = cfm_chord_cosine_diagnostics(
        node_current=node_current,
        node_target=node_target,
        node_endpoint=node_endpoint,
        edge_current=edge_current,
        edge_target=edge_target,
        edge_endpoint=edge_endpoint,
        t=torch.tensor([0.25]),
    )

    assert state["node_cos_v_theta_chord"].item() == pytest.approx(0.0)
    assert state["edge_cos_v_theta_chord"].item() == pytest.approx(1.0)
    assert state["cos_v_theta_chord"].item() == pytest.approx(8.0 / torch.sqrt(torch.tensor(85.0)).item())
