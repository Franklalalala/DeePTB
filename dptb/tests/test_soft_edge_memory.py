from __future__ import annotations

import pytest
import torch

from dptb.nn.embedding.soft_edge_memory import EquivariantSoftEdgeMemory


IRREPS = "3x0e + 2x1o + 1x0o + 1x2e"


def test_zero_initialized_memory_is_exact_noop_and_attention_is_soft():
    torch.manual_seed(7)
    module = EquivariantSoftEdgeMemory(
        IRREPS,
        num_slots=11,
        num_heads=2,
        head_dim=5,
        gate_mode="deepseek",
        zero_init_output=True,
    )
    features = torch.randn(6, module.irreps.dim, requires_grad=True)
    output, diagnostics = module(features, return_attention=True)

    assert torch.equal(output, features)
    attention = diagnostics["attention"]
    torch.testing.assert_close(
        attention.sum(dim=-1), torch.ones_like(attention[..., 0])
    )
    assert torch.all(attention > 0)
    assert torch.isfinite(diagnostics["attention_entropy"])
    assert torch.isfinite(diagnostics["gate_mean"])

    output.square().mean().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert module.output.weight.grad is not None


def test_memory_updates_only_invariant_even_scalars():
    torch.manual_seed(13)
    module = EquivariantSoftEdgeMemory(
        IRREPS,
        num_slots=7,
        num_heads=2,
        head_dim=4,
        gate_mode="deepseek",
        zero_init_output=False,
    )
    features = torch.randn(8, module.irreps.dim)
    output, _ = module(features)
    scalar_indices = module.scalar_indices.cpu()
    non_scalar_mask = torch.ones(module.irreps.dim, dtype=torch.bool)
    non_scalar_mask[scalar_indices] = False

    assert not torch.equal(output[:, scalar_indices], features[:, scalar_indices])
    assert torch.equal(output[:, non_scalar_mask], features[:, non_scalar_mask])


def test_production_mode_skips_unused_diagnostics():
    module = EquivariantSoftEdgeMemory(
        IRREPS,
        num_slots=5,
        diagnostics_mode="off",
    )
    _, diagnostics = module(torch.randn(4, module.irreps.dim))
    assert diagnostics == {}


@pytest.mark.parametrize("gate_mode", ["deepseek", "linear"])
def test_memory_empty_edges_and_non_soc_guards(gate_mode):
    module = EquivariantSoftEdgeMemory(
        IRREPS,
        num_slots=5,
        gate_mode=gate_mode,
        validate_inputs=True,
    )
    empty = torch.empty(0, module.irreps.dim)
    output, diagnostics = module(empty, return_attention=True)
    assert output.shape == empty.shape
    assert diagnostics["attention"].shape == (0, module.num_heads, module.num_slots)
    assert diagnostics["gate_mean"].item() == 0.0

    with pytest.raises(TypeError, match="non-SOC"):
        module(torch.zeros(2, module.irreps.dim, dtype=torch.complex64))
    with pytest.raises(ValueError, match="NaN"):
        bad = torch.zeros(2, module.irreps.dim)
        bad[0, 0] = torch.nan
        module(bad)


def test_memory_rejects_invalid_gate_mode():
    with pytest.raises(ValueError, match="gate_mode"):
        EquivariantSoftEdgeMemory(IRREPS, gate_mode="hash")


def test_single_scalar_input_norm_keeps_memory_conditional():
    torch.manual_seed(29)
    module = EquivariantSoftEdgeMemory(
        "1x0e + 1x1o",
        num_slots=5,
        num_heads=1,
        head_dim=3,
        zero_init_output=False,
        use_input_norm=True,
    )
    features = torch.zeros(2, module.irreps.dim)
    features[:, 0] = torch.tensor([-1.0, 2.0])
    _, diagnostics = module(features, return_attention=True)
    assert not torch.allclose(
        diagnostics["attention"][0], diagnostics["attention"][1]
    )


def test_zero_init_has_explicit_one_step_gradient_delay_then_starts():
    torch.manual_seed(31)
    module = EquivariantSoftEdgeMemory(
        IRREPS,
        num_slots=7,
        num_heads=2,
        head_dim=4,
        zero_init_output=True,
    )
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    features = torch.randn(9, module.irreps.dim)

    optimizer.zero_grad(set_to_none=True)
    first, _ = module(features)
    first.square().mean().backward()
    assert module.output.weight.grad is not None
    assert module.output.weight.grad.abs().sum() > 0
    for parameter in (
        module.query.weight,
        module.keys,
        module.values,
        module.gate_query_weight,
        module.gate_key_weight,
    ):
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad) == 0
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)
    second, _ = module(features)
    second.square().mean().backward()
    for parameter in (
        module.query.weight,
        module.keys,
        module.values,
        module.gate_query_weight,
        module.gate_key_weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
