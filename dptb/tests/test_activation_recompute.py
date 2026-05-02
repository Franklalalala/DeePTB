import torch

from dptb.nn.activation_recompute import (
    checkpoint_module_call,
    checkpoint_so2_linear_call,
    configure_activation_recompute,
)
from dptb.nn.tensor_product_moe_v3 import MOLEGlobals


class CountingBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, scale, optional_state=None):
        self.calls += 1
        y = torch.sin(x) * scale
        return y, optional_state


def test_checkpoint_module_call_recomputes_during_backward():
    block = CountingBlock()
    block.train()
    x = torch.randn(8, requires_grad=True)
    scale = torch.tensor(2.0)

    y, optional_state = checkpoint_module_call(
        block,
        x,
        scale,
        None,
        enabled=True,
        use_reentrant=False,
    )

    assert optional_state is None
    assert block.calls == 1

    y.sum().backward()

    assert block.calls == 2
    assert x.grad is not None


def test_checkpoint_module_call_disabled_uses_plain_forward():
    block = CountingBlock()
    block.train()
    x = torch.randn(8, requires_grad=True)

    y, _ = checkpoint_module_call(block, x, torch.tensor(2.0), None, enabled=False)
    y.sum().backward()

    assert block.calls == 1
    assert x.grad is not None


class FakeSO2Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, edge_vector, mole_globals, latents=None, wigner_D_all=None):
        self.calls += 1
        scale = mole_globals.coefficients.sum()
        out = x * scale + edge_vector.sum() * 0.0
        if latents is not None:
            out = out + latents.sum(dim=-1, keepdim=True) * 0.0
        return out, wigner_D_all


def test_checkpoint_so2_linear_call_keeps_router_coefficient_grad():
    module = FakeSO2Linear()
    module.train()
    x = torch.randn(5, 3, requires_grad=True)
    edge_vector = torch.randn(5, 3)
    coefficients = torch.randn(2, 4, requires_grad=True)
    sizes = torch.tensor([2, 3], dtype=torch.long)
    mole_globals = MOLEGlobals(coefficients=coefficients, sizes=sizes)

    out, wigner_D_all = checkpoint_so2_linear_call(
        module,
        x,
        edge_vector,
        mole_globals,
        latents=None,
        wigner_D_all=None,
        enabled=True,
        use_reentrant=False,
    )

    assert wigner_D_all is None
    assert module.calls == 1

    out.sum().backward()

    assert module.calls == 2
    assert x.grad is not None
    assert coefficients.grad is not None


class UpdateNode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tp = torch.nn.Linear(1, 1)


class UpdateEdge(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tp = torch.nn.Linear(1, 1)


class OtherBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tp = torch.nn.Linear(1, 1)


def test_configure_activation_recompute_marks_tp_targets_without_state_dict_churn():
    model = torch.nn.Module()
    model.node = UpdateNode()
    model.edge = UpdateEdge()
    model.other = OtherBlock()
    before_keys = list(model.state_dict().keys())

    state = configure_activation_recompute(
        model,
        {
            "enabled": True,
            "targets": ["lem_moe_v3_tp"],
            "checkpoint_node_tp": True,
            "checkpoint_edge_tp": True,
            "use_reentrant": False,
            "preserve_rng_state": True,
        },
    )

    assert state == {"enabled": 1, "node_tp": 1, "edge_tp": 1}
    assert list(model.state_dict().keys()) == before_keys
    assert model.node._activation_recompute_enabled is True
    assert model.edge._activation_recompute_enabled is True
    assert not hasattr(model.other, "_activation_recompute_enabled")
