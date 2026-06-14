import pytest
import torch

from dptb.nn.embedding.flow_time import FlowTimeConditioner, sinusoidal_time_embedding


def test_sinusoidal_time_embedding_matches_qhflow2_shape_and_is_time_sensitive():
    t = torch.tensor([0.0, 0.5])
    emb = sinusoidal_time_embedding(t, embedding_dim=4, max_positions=2000)

    assert emb.shape == (2, 4)
    assert not torch.allclose(emb[0], emb[1])


def test_conditioner_maps_per_graph_time_to_nodes_and_only_changes_scalar_channels():
    conditioner = FlowTimeConditioner(
        scalar_channels=4,
        flow_time_key="flow_time",
        max_positions=2000,
    )
    node_features = torch.zeros(3, 9)
    data = {
        "batch": torch.tensor([0, 0, 1], dtype=torch.long),
        "flow_time": torch.tensor([0.0, 0.5]),
    }

    conditioned = conditioner(node_features, data)

    assert conditioned.shape == node_features.shape
    assert torch.allclose(conditioned[0], conditioned[1])
    assert not torch.allclose(conditioned[0], conditioned[2])
    assert torch.count_nonzero(conditioned[:, 4:]) == 0


def test_conditioner_requires_one_time_per_graph():
    conditioner = FlowTimeConditioner(scalar_channels=4)
    data = {
        "batch": torch.tensor([0, 0, 1], dtype=torch.long),
        "flow_time": torch.tensor([0.25]),
    }

    with pytest.raises(ValueError, match="one value per graph"):
        conditioner(torch.zeros(3, 4), data)


def test_conditioner_can_use_two_time_meanflow_channels():
    conditioner = FlowTimeConditioner(
        scalar_channels=4,
        flow_time_key="flow_time",
        flow_time_keys=("flow_time_t", "flow_time_r", "flow_time_h"),
        max_positions=2000,
    )
    node_features = torch.zeros(2, 4)
    base = {
        "batch": torch.tensor([0, 1], dtype=torch.long),
        "flow_time": torch.tensor([0.8, 0.8]),
        "flow_time_t": torch.tensor([0.8, 0.8]),
        "flow_time_r": torch.tensor([0.2, 0.6]),
        "flow_time_h": torch.tensor([0.6, 0.2]),
    }

    conditioned = conditioner(node_features, base)

    assert conditioned.shape == node_features.shape
    assert not torch.allclose(conditioned[0], conditioned[1])
