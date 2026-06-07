import copy
from pathlib import Path

import pytest


def test_lem_moe_v3_exposes_post_activation_expert_mixing_config():
    root = Path(__file__).resolve().parents[1]
    lem_source = (root / "nn" / "embedding" / "lem_moe_v3.py").read_text(encoding="utf-8")
    argcheck_source = (root / "utils" / "argcheck.py").read_text(encoding="utf-8")

    assert 'so2_expert_mixing_mode: str = "pre_activation"' in lem_source
    assert "SO2PostActivationExpertMixer" in lem_source
    assert "post_activation_expert_mixer" in lem_source
    assert 'Argument("so2_expert_mixing_mode", str' in argcheck_source
    assert 'Argument("so2_expert_route_chunk_size", [int, None]' in argcheck_source
    assert 'Argument("so2_expert_route_checkpoint", bool' in argcheck_source


def _manual_expert_linear(torch, layer, x, expert_index):
    flat_x = x.reshape(-1, layer.in_features)
    flat_index = expert_index.reshape(-1).to(device=x.device, dtype=torch.long)
    if x.ndim > 2 and expert_index.ndim == 1:
        flat_index = expert_index.reshape(-1, *([1] * (x.ndim - 2))).expand(x.shape[:-1]).reshape(-1)

    weight = layer.weight_experts.index_select(0, flat_index)
    out = torch.bmm(weight, flat_x.unsqueeze(-1)).squeeze(-1)
    if layer.bias_experts is not None:
        out = out + layer.bias_experts.index_select(0, flat_index)
    return out.reshape(*x.shape[:-1], layer.out_features)


@pytest.mark.parametrize("leading_shape", [(7,), (5, 2)])
def test_mole_linear_apply_experts_matches_manual_forward_and_grad(leading_shape):
    torch = pytest.importorskip("torch")
    from dptb.nn.tensor_product_moe_v3 import MOLELinear

    torch.manual_seed(20260604)
    dtype = torch.float64
    num_experts = 3
    in_features = 4
    out_features = 5

    layer = MOLELinear(
        in_features,
        out_features,
        num_experts=num_experts,
        num_shared_experts=0,
        bias=True,
        mole_linear_mode="indexed_ref",
    ).to(dtype=dtype)
    ref_layer = copy.deepcopy(layer)

    x = torch.randn(*leading_shape, in_features, dtype=dtype, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    expert_index = torch.tensor([2, 0, 1, 2, 1, 0, 2][: leading_shape[0]], dtype=torch.long)

    out = layer.apply_experts(x, expert_index)
    ref = _manual_expert_linear(torch, ref_layer, x_ref, expert_index)

    torch.testing.assert_close(out, ref, atol=1e-12, rtol=1e-12)

    out.square().sum().backward()
    ref.square().sum().backward()

    torch.testing.assert_close(x.grad, x_ref.grad, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        layer.weight_experts.grad,
        ref_layer.weight_experts.grad,
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        layer.bias_experts.grad,
        ref_layer.bias_experts.grad,
        atol=1e-12,
        rtol=1e-12,
    )


def test_so2_forward_expert_routes_matches_onehot_reference_and_backward():
    torch = pytest.importorskip("torch")
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, SO2_Linear

    torch.manual_seed(20260604)
    dtype = torch.float64
    num_experts = 3
    n_rows = 6
    layer = SO2_Linear(
        irreps_in="2x0e + 1x1o",
        irreps_out="2x0e + 1x1o",
        radial_emb=False,
        num_experts=num_experts,
        num_shared_experts=0,
        rotate_in=True,
        rotate_out=True,
        mole_linear_mode="indexed_ref",
        so2_fusion_mode="streamed_m_major_ref",
    ).to(dtype=dtype)

    x = torch.randn(n_rows, layer.irreps_in.dim, dtype=dtype, requires_grad=True)
    R = torch.randn(n_rows, 3, dtype=dtype)
    expert_index = torch.tensor([0, 2, 1, 2, 0, 1], dtype=torch.long)
    coeffs = torch.nn.functional.one_hot(expert_index, num_classes=num_experts).to(dtype=dtype)
    onehot_globals = MOLEGlobals(
        coefficients=coeffs,
        graph_index=torch.arange(n_rows, dtype=torch.long),
    )

    ref, _ = layer(x, R, onehot_globals)
    out, _ = layer.forward_expert_routes(x, R, expert_index)

    torch.testing.assert_close(out, ref, atol=1e-10, rtol=1e-10)

    loss = out.square().mean()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    assert all(
        param.grad is None or torch.isfinite(param.grad).all()
        for param in layer.parameters()
    )


def test_post_activation_expert_mixer_matches_chunked_reference_and_not_prefusion():
    torch = pytest.importorskip("torch")
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, SO2PostActivationExpertMixer, SO2_Linear

    torch.manual_seed(20260604)
    dtype = torch.float64
    num_experts = 2
    n_rows = 5
    layer = SO2_Linear(
        irreps_in="2x0e + 1x1o",
        irreps_out="2x0e + 1x1o",
        radial_emb=False,
        num_experts=num_experts,
        num_shared_experts=0,
        rotate_in=True,
        rotate_out=True,
        mole_linear_mode="indexed_ref",
        so2_fusion_mode="streamed_m_major_ref",
    ).to(dtype=dtype)
    activation = torch.nn.SiLU()
    router = torch.nn.Linear(2, 1, bias=False).to(dtype=dtype)
    with torch.no_grad():
        router.weight.zero_()

    mixer = SO2PostActivationExpertMixer(
        tp=layer,
        activation=activation,
        router_from_0e=router,
        scalar_dim=2,
        route_chunk_size=2,
        checkpoint_routes=True,
    )

    x = torch.randn(n_rows, layer.irreps_in.dim, dtype=dtype, requires_grad=True)
    R = torch.randn(n_rows, 3, dtype=dtype)
    coeffs = torch.full((1, num_experts), 0.5, dtype=dtype)
    topk_indices = torch.tensor([[0, 1]], dtype=torch.long)
    topk_values = torch.full((1, num_experts), 0.5, dtype=dtype)
    mole_globals = MOLEGlobals(
        coefficients=coeffs,
        sizes=torch.tensor([n_rows], dtype=torch.long),
        topk_indices=topk_indices,
        topk_values=topk_values,
    )

    out, _ = mixer(x, R, mole_globals)

    expert_outputs = []
    for expert_id in range(num_experts):
        expert_index = torch.full((n_rows,), expert_id, dtype=torch.long)
        y_e, _ = layer.forward_expert_routes(x, R, expert_index)
        expert_outputs.append(activation(y_e))
    ref = torch.stack(expert_outputs, dim=1).mean(dim=1)
    torch.testing.assert_close(out, ref, atol=1e-10, rtol=1e-10)

    prefused, _ = layer(x, R, mole_globals)
    prefused = activation(prefused)
    assert (out - prefused).abs().max().item() > 1e-8

    out.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(
        param.grad is None or torch.isfinite(param.grad).all()
        for param in list(layer.parameters()) + list(router.parameters())
    )
