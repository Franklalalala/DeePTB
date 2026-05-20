import torch


def _linear_expected(x, weights, bias):
    out = torch.einsum("...i,oi->...o", x, weights)
    if bias is not None:
        out = out + bias
    return out


def test_mole_linear_preserves_graph_level_coefficients():
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLELinear

    layer = MOLELinear(2, 1, num_experts=3, num_shared_experts=1, bias=True)
    with torch.no_grad():
        layer.weight_experts.copy_(
            torch.tensor([[[1.0, 0.0]], [[0.0, 2.0]], [[-1.0, 1.0]]])
        )
        layer.bias_experts.copy_(torch.tensor([[0.1], [0.2], [0.3]]))
        layer.weight_shared.copy_(torch.tensor([[[0.5, 0.5]]]))
        layer.bias_shared.copy_(torch.tensor([[0.05]]))

    x = torch.tensor([[2.0, 4.0], [1.0, 3.0], [5.0, 7.0]])
    coeffs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.25, 0.75]])
    globals_ = MOLEGlobals(coefficients=coeffs, sizes=torch.tensor([2, 1]))

    out = layer(x, globals_)

    expected_parts = []
    for x_sys, c in zip(torch.split(x, [2, 1], dim=0), coeffs):
        mixed_w = torch.einsum("e,eoi->oi", c, layer.weight_experts)
        mixed_b = torch.einsum("e,eo->o", c, layer.bias_experts)
        mixed_w = mixed_w + layer.weight_shared.sum(0)
        mixed_b = mixed_b + layer.bias_shared.sum(0)
        expected_parts.append(_linear_expected(x_sys, mixed_w, mixed_b))
    expected = torch.cat(expected_parts, dim=0)

    torch.testing.assert_close(out, expected)


def test_mole_linear_accepts_edge_level_coefficients_without_sizes():
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLELinear

    layer = MOLELinear(2, 1, num_experts=3, num_shared_experts=1, bias=True)
    with torch.no_grad():
        layer.weight_experts.copy_(
            torch.tensor([[[1.0, 0.0]], [[0.0, 2.0]], [[-1.0, 1.0]]])
        )
        layer.bias_experts.copy_(torch.tensor([[0.1], [0.2], [0.3]]))
        layer.weight_shared.copy_(torch.tensor([[[0.5, 0.5]]]))
        layer.bias_shared.copy_(torch.tensor([[0.05]]))

    x = torch.tensor([[2.0, 4.0], [1.0, 3.0]])
    coeffs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.25, 0.75]])
    globals_ = MOLEGlobals(coefficients=coeffs)

    out = layer(x, globals_)

    expected = []
    shared_w = layer.weight_shared.sum(0)
    shared_b = layer.bias_shared.sum(0)
    for x_edge, c in zip(x, coeffs):
        routed_w = torch.einsum("e,eoi->oi", c, layer.weight_experts)
        routed_b = torch.einsum("e,eo->o", c, layer.bias_experts)
        expected.append(_linear_expected(x_edge, routed_w + shared_w, routed_b + shared_b))
    expected = torch.stack(expected, dim=0)

    torch.testing.assert_close(out, expected)


def test_mole_linear_edge_level_coefficients_support_extra_batch_dims():
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLELinear

    layer = MOLELinear(2, 1, num_experts=2, num_shared_experts=0, bias=False)
    with torch.no_grad():
        layer.weight_experts.copy_(torch.tensor([[[1.0, 0.0]], [[0.0, 2.0]]]))

    x = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    coeffs = torch.tensor([[1.0, 0.0], [0.25, 0.75]])
    globals_ = MOLEGlobals(coefficients=coeffs)

    out = layer(x, globals_)

    expected = []
    for x_edge, c in zip(x, coeffs):
        routed_w = torch.einsum("e,eoi->oi", c, layer.weight_experts)
        expected.append(_linear_expected(x_edge, routed_w, None))
    expected = torch.stack(expected, dim=0)

    torch.testing.assert_close(out, expected)


def test_edge_moe_embedding_registers_separately_from_global_moe():
    from dptb.nn.embedding.emb import Embedding
    from dptb.nn.embedding.lem_moe_v3 import LemMoEV3
    from dptb.nn.embedding.lem_moe_v3_edge import LemMoEV3Edge, LemMoEV3EdgeH0

    assert Embedding._register["lem_moe_v3"] is LemMoEV3
    assert Embedding._register["lem_moe_v3_edge"] is LemMoEV3Edge
    assert Embedding._register["lem_moe_v3_edge_h0"] is LemMoEV3EdgeH0
