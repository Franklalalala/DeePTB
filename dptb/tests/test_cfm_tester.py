import torch
from torch import nn

from dptb.nnops.flow import HamiltonianCFM
from dptb.nnops import tester as tester_mod
from dptb.nnops.tester import Tester


class _FakeBatch:
    def __init__(self, data):
        self.data = data
        self.__slices__ = {}
        self.__cumsum__ = {}
        self.__cat_dims__ = {}
        self.__num_nodes_list__ = []
        self.__data_class__ = _FakeBatch

    def to(self, device):
        return self


class _EndpointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.grad_enabled = []

    def forward(self, data):
        self.grad_enabled.append(torch.is_grad_enabled())
        out = data.copy()
        out["node_features"] = torch.full_like(data["node_h0"], 2.0)
        out["edge_features"] = torch.full_like(data["edge_h0"], 4.0)
        return out


class _MSELoss:
    last_onsite_loss = None
    last_hopping_loss = None

    def __call__(self, pred, ref):
        self.last_onsite_loss = (pred["node_features"] - ref["node_features"]).square().mean()
        self.last_hopping_loss = (pred["edge_features"] - ref["edge_features"]).square().mean()
        return self.last_onsite_loss + self.last_hopping_loss


def test_tester_reports_cfm_sampling_metrics_instead_of_using_direct_loss(monkeypatch):
    raw = {
        "batch": torch.tensor([0, 0]),
        "edge_index": torch.tensor([[0], [1]]),
        "node_h0": torch.zeros(2, 1),
        "edge_h0": torch.zeros(1, 1),
        "node_features": torch.full((2, 1), 2.0),
        "edge_features": torch.full((1, 1), 4.0),
    }
    monkeypatch.setattr(tester_mod.AtomicData, "to_AtomicDataDict", lambda batch: batch.data)
    observed = []
    tester = Tester.__new__(Tester)
    tester.model = _EndpointModel()
    tester.device = torch.device("cpu")
    tester.test_lossfunc = _MSELoss()
    tester.flow_cfm = HamiltonianCFM(
        {
            "enabled": True,
            "strict_h0": True,
            "validation_ode_steps": [1, 3],
        }
    )
    tester.log_direct_target_fed_loss = True
    tester.iter = 0
    tester.call_plugins = lambda queue_name, time, **state: observed.append(state)

    loss = tester.iteration(_FakeBatch(raw))

    assert loss.item() == 0.0
    assert observed[0]["test_loss"].item() == 0.0
    assert observed[0]["test_direct_target_fed_loss"].item() == 0.0
    assert observed[0]["test_cfm_euler_1_loss"].item() == 0.0
    assert observed[0]["test_cfm_euler_3_loss"].item() == 0.0
    assert tester.model.grad_enabled
    assert not any(tester.model.grad_enabled)
