"""Small orbital masks and labelled graphs for structured prior noise tests."""
import torch
from e3nn import o3


class FakeIDP:
    """Minimal mapper with scalar/vector spans and two distinct type masks."""

    def __init__(self, *, device):
        self.orbpair_irreps = o3.Irreps("1x0e+1x1o")
        self.mask_to_nrme = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]],
                                         device=device, dtype=torch.bool)
        self.mask_to_erme = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 0]],
                                         device=device, dtype=torch.bool)


def make_batch(*, device, dtype):
    node_base = torch.arange(12, device=device, dtype=dtype).reshape(3, 4) / 100.0
    edge_base = torch.arange(16, device=device, dtype=dtype).reshape(4, 4) / 100.0
    node_target = node_base + torch.tensor([[1., 2., 3., 4.], [2., 0., 0., 0.],
                                           [1.5, 2.5, 3.5, 4.5]], device=device, dtype=dtype)
    edge_target = edge_base + torch.tensor([[1., 2., 3., 4.], [0., 2., 3., 0.],
                                           [1.5, 2.5, 3.5, 4.5], [0., 1., 2., 0.]],
                                          device=device, dtype=dtype)
    data = {
        "node_h0": node_base.clone(), "edge_h0": edge_base.clone(),
        "edge_index": torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], device=device),
        "batch": torch.tensor([0, 0, 1], device=device),
        "atom_types": torch.tensor([0, 1, 0], device=device),
        "edge_type": torch.tensor([0, 1, 0, 1], device=device),
        "node_features": node_base.clone(), "edge_features": edge_base.clone(),
    }
    return data, {"node_features": node_target, "edge_features": edge_target}
