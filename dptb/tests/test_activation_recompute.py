import torch
import pytest

from dptb.nn.activation_recompute import (
    checkpoint_so2_linear_from_parts,
    checkpoint_module_call,
    checkpoint_so2_linear_call,
    clone_mole_globals_for_recompute,
    configure_activation_recompute,
)
from dptb.nn.tensor_product_moe_v3 import MOLEGlobals
from dptb.nn.tensor_product_moe_v3 import SO2_Linear


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


def test_checkpoint_module_call_rejects_reentrant():
    block = CountingBlock()
    block.train()
    x = torch.randn(8, requires_grad=True)

    with pytest.raises(ValueError, match="use_reentrant=True is not supported"):
        checkpoint_module_call(
            block,
            x,
            torch.tensor(2.0),
            None,
            enabled=True,
            use_reentrant=True,
        )


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


class RecordingSO2Linear(FakeSO2Linear):
    def __init__(self):
        super().__init__()
        self.seen_shapes = []
        self.seen_graph_indices = []

    def forward(self, x, edge_vector, mole_globals, latents=None, wigner_D_all=None):
        self.seen_shapes.append(tuple(x.shape))
        self.seen_graph_indices.append(getattr(mole_globals, "graph_index", None))
        return super().forward(x, edge_vector, mole_globals, latents, wigner_D_all)


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


def test_checkpoint_so2_linear_from_parts_recomputes_gather_cat_and_keeps_grads():
    module = RecordingSO2Linear()
    module.train()
    node_in = torch.randn(4, 2, requires_grad=True)
    edge_in = torch.randn(3, 1, requires_grad=True)
    edge_center = torch.tensor([0, 1, 2, 3, 1], dtype=torch.long)
    edge_neighbor = torch.tensor([1, 2, 3, 0, 2], dtype=torch.long)
    active_edges = torch.tensor([0, 2, 4], dtype=torch.long)
    edge_vector = torch.randn(5, 3, requires_grad=True)
    latents = torch.randn(5, 2, requires_grad=True)
    coefficients = torch.randn(2, 4, requires_grad=True)
    sizes = torch.tensor([1, 2], dtype=torch.long)
    graph_index = torch.tensor([0, 1, 1], dtype=torch.long)
    mole_globals = MOLEGlobals(
        coefficients=coefficients,
        sizes=sizes,
        graph_index=graph_index,
    )

    out, _ = checkpoint_so2_linear_from_parts(
        module,
        node_in,
        edge_in,
        edge_center,
        active_edges,
        edge_vector,
        mole_globals,
        latents=latents,
        wigner_D_all=None,
        edge_neighbor=edge_neighbor,
        enabled=True,
    )

    assert module.calls == 1
    assert module.seen_shapes == [(3, 5)]

    out.sum().backward()

    assert module.calls == 2
    assert module.seen_shapes == [(3, 5), (3, 5)]
    assert node_in.grad is not None
    assert edge_in.grad is not None
    assert edge_vector.grad is not None
    assert latents.grad is not None
    assert coefficients.grad is not None


def test_checkpoint_so2_linear_actual_compact_blocks_matches_plain_forward_and_grads():
    torch.manual_seed(0)
    module = SO2_Linear(
        "1x1e",
        "1x1e",
        num_experts=2,
        num_shared_experts=1,
        rotate_in=True,
        rotate_out=True,
        wigner_apply_mode="compact_blocks",
    )
    module.train()
    x_base = torch.randn(4, 3)
    edge_vector_base = torch.randn(4, 3)
    coefficients_base = torch.randn(2, 2)
    sizes = torch.tensor([2, 2], dtype=torch.long)

    def _run(enabled):
        for param in module.parameters():
            param.grad = None
        x = x_base.clone().requires_grad_(True)
        edge_vector = edge_vector_base.clone().requires_grad_(True)
        coefficients = coefficients_base.clone().requires_grad_(True)
        mole_globals = MOLEGlobals(coefficients=coefficients, sizes=sizes)
        out, _ = checkpoint_so2_linear_call(
            module,
            x,
            edge_vector,
            mole_globals,
            enabled=enabled,
        )
        loss = out.square().sum()
        loss.backward()
        param_grads = [
            param.grad.detach().clone()
            for param in module.parameters()
            if param.grad is not None
        ]
        return (
            out.detach(),
            x.grad.detach(),
            edge_vector.grad.detach(),
            coefficients.grad.detach(),
            param_grads,
        )

    plain = _run(False)
    recomputed = _run(True)

    for lhs, rhs in zip(plain[:4], recomputed[:4]):
        torch.testing.assert_close(lhs, rhs, rtol=1e-5, atol=1e-6)
    assert len(plain[4]) == len(recomputed[4])
    for lhs, rhs in zip(plain[4], recomputed[4]):
        torch.testing.assert_close(lhs, rhs, rtol=1e-5, atol=1e-6)


def test_clone_mole_globals_for_recompute_preserves_graph_index_with_split_sizes():
    coefficients = torch.randn(2, 4, requires_grad=True)
    sizes = torch.tensor([2, 3], dtype=torch.long)
    graph_index = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    src = MOLEGlobals(coefficients=coefficients, split_sizes=(2, 3))

    cloned = clone_mole_globals_for_recompute(
        src,
        coefficients=coefficients,
        sizes=sizes,
        split_sizes=(2, 3),
        graph_index=graph_index,
    )

    assert cloned.split_sizes == (2, 3)
    assert cloned.graph_index is graph_index


def test_clone_mole_globals_for_recompute_preserves_graph_index_without_hot_path_validation():
    coefficients = torch.randn(2, 4, requires_grad=True)
    sizes = torch.tensor([2, 3], dtype=torch.long)
    graph_index = torch.tensor([0, 1, 0, 1, 1], dtype=torch.long)
    src = MOLEGlobals(coefficients=coefficients, split_sizes=(2, 3))

    cloned = clone_mole_globals_for_recompute(
        src,
        coefficients=coefficients,
        sizes=sizes,
        split_sizes=(2, 3),
        graph_index=graph_index,
    )

    assert cloned.split_sizes == (2, 3)
    assert cloned.graph_index is graph_index


def test_clone_mole_globals_for_recompute_validates_graph_index_only_when_enabled(monkeypatch):
    coefficients = torch.randn(2, 4, requires_grad=True)
    sizes = torch.tensor([2, 3], dtype=torch.long)
    graph_index = torch.tensor([0, 1, 0, 1, 1], dtype=torch.long)
    src = MOLEGlobals(coefficients=coefficients, split_sizes=(2, 3))

    monkeypatch.setenv("DPTB_ACTIVATION_RECOMPUTE_VALIDATE_GRAPH_INDEX", "1")

    with pytest.raises(ValueError, match="graph_index is inconsistent"):
        clone_mole_globals_for_recompute(
            src,
            coefficients=coefficients,
            sizes=sizes,
            split_sizes=(2, 3),
            graph_index=graph_index,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for cueq_indexed_linear")
def test_checkpoint_so2_linear_actual_cueq_indexed_matches_plain_forward_and_grads():
    pytest.importorskip("cuequivariance")
    pytest.importorskip("cuequivariance_torch")

    torch.manual_seed(0)
    device = torch.device("cuda")
    module = SO2_Linear(
        "1x1e",
        "1x1e",
        num_experts=2,
        num_shared_experts=1,
        rotate_in=True,
        rotate_out=True,
        wigner_apply_mode="compact_blocks",
        mole_linear_mode="cueq_indexed_linear",
    ).to(device)
    module.train()

    x_base = torch.randn(4, 3, device=device)
    edge_vector_base = torch.randn(4, 3, device=device)
    coefficients_base = torch.randn(2, 2, device=device)
    sizes = torch.tensor([2, 2], dtype=torch.long, device=device)
    graph_index = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=device)

    def _run(enabled):
        module.zero_grad(set_to_none=True)
        x = x_base.clone().requires_grad_(True)
        edge_vector = edge_vector_base.clone().requires_grad_(True)
        coefficients = coefficients_base.clone().requires_grad_(True)
        mole_globals = MOLEGlobals(
            coefficients=coefficients,
            sizes=sizes,
            graph_index=graph_index,
        )
        out, _ = checkpoint_so2_linear_call(
            module,
            x,
            edge_vector,
            mole_globals,
            enabled=enabled,
        )
        loss = out.square().sum()
        loss.backward()
        param_grads = [
            param.grad.detach().clone()
            for param in module.parameters()
            if param.grad is not None
        ]
        return (
            out.detach(),
            x.grad.detach(),
            edge_vector.grad.detach(),
            coefficients.grad.detach(),
            param_grads,
        )

    plain = _run(False)
    recomputed = _run(True)

    for lhs, rhs in zip(plain[:4], recomputed[:4]):
        torch.testing.assert_close(lhs, rhs, rtol=1e-5, atol=1e-6)
    assert len(plain[4]) == len(recomputed[4])
    for lhs, rhs in zip(plain[4], recomputed[4]):
        torch.testing.assert_close(lhs, rhs, rtol=1e-5, atol=1e-6)


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
        },
    )

    assert state == {"enabled": 1, "node_tp": 1, "edge_tp": 1}
    assert list(model.state_dict().keys()) == before_keys
    assert model.node._activation_recompute_enabled is True
    assert model.edge._activation_recompute_enabled is True
    assert model.node._activation_recompute_preserve_rng_state is False
    assert model.edge._activation_recompute_preserve_rng_state is False
    assert not hasattr(model.other, "_activation_recompute_enabled")


def test_configure_activation_recompute_rejects_reentrant():
    model = torch.nn.Module()
    model.node = UpdateNode()

    with pytest.raises(ValueError, match="use_reentrant=True is not supported"):
        configure_activation_recompute(
            model,
            {
                "enabled": True,
                "targets": ["lem_moe_v3_tp"],
                "use_reentrant": True,
            },
        )


def test_configure_activation_recompute_clears_stale_flags_when_disabled():
    model = torch.nn.Module()
    model.node = UpdateNode()
    model.edge = UpdateEdge()

    configure_activation_recompute(
        model,
        {
            "enabled": True,
            "targets": ["lem_moe_v3_tp"],
        },
    )
    assert hasattr(model.node, "_activation_recompute_enabled")
    assert hasattr(model.edge, "_activation_recompute_enabled")

    state = configure_activation_recompute(model, {"enabled": False})

    assert state == {"enabled": 0, "node_tp": 0, "edge_tp": 0}
    assert not hasattr(model.node, "_activation_recompute_enabled")
    assert not hasattr(model.edge, "_activation_recompute_enabled")
