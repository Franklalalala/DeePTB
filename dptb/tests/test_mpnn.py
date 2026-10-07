"""The upstream MPNN executes its message-passing layer on a small graph."""
import torch

from dptb.nn.embedding.mpnn import MPNN


def test_mpnn_message_passing_forward_and_backward():
    network = dict(neurons=[8], activation="tanh", if_batch_normalized=False)
    model = MPNN(r_max=4.0, p=6, n_basis=4, n_node=4, n_edge=4,
                 n_atom=1, n_layer=1, node_net=dict(network), edge_net=dict(network))
    edges = torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]])
    data = dict(pos=torch.tensor([[0., 0., 0.], [1.1, .2, -.1], [-.3, 1.2, .4]]),
                edge_index=edges, env_index=edges.clone(),
                atom_types=torch.zeros(3, 1, dtype=torch.long))
    result = model(data)
    assert result["node_features"].shape == (3, 4)
    assert result["edge_features"].shape == (6, 4)
    loss = result["node_features"].square().sum() + result["edge_features"].square().sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.layers[0].parameters())
