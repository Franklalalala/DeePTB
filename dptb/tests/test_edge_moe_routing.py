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


def test_mole_linear_accepts_compact_graph_index_coefficients():
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
    graph_index = torch.tensor([0, 1, 0])
    globals_ = MOLEGlobals(coefficients=coeffs, graph_index=graph_index)

    out = layer(x, globals_)

    expected = []
    shared_w = layer.weight_shared.sum(0)
    shared_b = layer.bias_shared.sum(0)
    for x_edge, coeff_idx in zip(x, graph_index):
        c = coeffs[coeff_idx]
        routed_w = torch.einsum("e,eoi->oi", c, layer.weight_experts)
        routed_b = torch.einsum("e,eo->o", c, layer.bias_experts)
        expected.append(_linear_expected(x_edge, routed_w + shared_w, routed_b + shared_b))
    expected = torch.stack(expected, dim=0)

    torch.testing.assert_close(out, expected)


def test_mole_linear_compact_graph_index_supports_extra_batch_dims():
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLELinear

    layer = MOLELinear(2, 1, num_experts=2, num_shared_experts=0, bias=False)
    with torch.no_grad():
        layer.weight_experts.copy_(torch.tensor([[[1.0, 0.0]], [[0.0, 2.0]]]))

    x = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ]
    )
    coeffs = torch.tensor([[1.0, 0.0], [0.25, 0.75]])
    graph_index = torch.tensor([0, 1, 0])
    globals_ = MOLEGlobals(coefficients=coeffs, graph_index=graph_index)

    out = layer(x, globals_)

    expected = []
    for x_edge, coeff_idx in zip(x, graph_index):
        routed_w = torch.einsum("e,eoi->oi", coeffs[coeff_idx], layer.weight_experts)
        expected.append(_linear_expected(x_edge, routed_w, None))
    expected = torch.stack(expected, dim=0)

    torch.testing.assert_close(out, expected)


def test_mole_linear_grouped_compact_graph_index_matches_expected():
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLELinear

    layer = MOLELinear(2, 1, num_experts=3, num_shared_experts=1, bias=True)
    with torch.no_grad():
        layer.weight_experts.copy_(
            torch.tensor([[[1.0, 0.0]], [[0.0, 2.0]], [[-1.0, 1.0]]])
        )
        layer.bias_experts.copy_(torch.tensor([[0.1], [0.2], [0.3]]))
        layer.weight_shared.copy_(torch.tensor([[[0.5, 0.5]]]))
        layer.bias_shared.copy_(torch.tensor([[0.05]]))

    x = torch.tensor([[2.0, 4.0], [1.0, 3.0], [5.0, 7.0], [11.0, 13.0]])
    coeffs = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.25, 0.75],
            [0.2, 0.3, 0.5],
        ]
    )
    graph_index = torch.tensor([0, 2, 1, 2])
    globals_ = MOLEGlobals(coefficients=coeffs, graph_index=graph_index)

    out = layer._forward_indexed_grouped(x, globals_)

    expected = []
    shared_w = layer.weight_shared.sum(0)
    shared_b = layer.bias_shared.sum(0)
    for x_edge, coeff_idx in zip(x, graph_index):
        c = coeffs[coeff_idx]
        routed_w = torch.einsum("e,eoi->oi", c, layer.weight_experts)
        routed_b = torch.einsum("e,eo->o", c, layer.bias_experts)
        expected.append(_linear_expected(x_edge, routed_w + shared_w, routed_b + shared_b))
    expected = torch.stack(expected, dim=0)

    torch.testing.assert_close(out, expected)


def test_mole_globals_caches_indexed_permutation_for_compact_groups():
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals

    graph_index = torch.tensor([2, 0, 2, 1, 0])
    globals_ = MOLEGlobals(
        coefficients=torch.zeros(3, 2),
        graph_index=graph_index,
    )

    permute_idx, unpermute_idx, group_offsets, sorted_graph_index = globals_.indexed_permutation()
    cached = globals_.indexed_permutation()

    assert sorted_graph_index.tolist() == [0, 0, 1, 2, 2]
    assert graph_index.index_select(0, permute_idx).tolist() == [0, 0, 1, 2, 2]
    assert torch.arange(graph_index.numel()).index_select(0, permute_idx).index_select(0, unpermute_idx).tolist() == list(
        range(graph_index.numel())
    )
    assert group_offsets.tolist() == [0, 2, 3, 5]
    assert cached[0] is permute_idx
    assert cached[1] is unpermute_idx
    assert cached[2] is group_offsets
    assert cached[3] is sorted_graph_index


def test_mole_linear_compact_graph_index_uses_contiguous_grouped_path():
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLELinear

    layer = MOLELinear(2, 1, num_experts=3, num_shared_experts=1, bias=True)
    with torch.no_grad():
        layer.weight_experts.copy_(
            torch.tensor([[[1.0, 0.0]], [[0.0, 2.0]], [[-1.0, 1.0]]])
        )
        layer.bias_experts.copy_(torch.tensor([[0.1], [0.2], [0.3]]))
        layer.weight_shared.copy_(torch.tensor([[[0.5, 0.5]]]))
        layer.bias_shared.copy_(torch.tensor([[0.05]]))

    x = torch.tensor(
        [
            [[2.0, 4.0], [3.0, 5.0]],
            [[1.0, 3.0], [2.0, 4.0]],
            [[5.0, 7.0], [6.0, 8.0]],
            [[11.0, 13.0], [12.0, 14.0]],
        ]
    )
    coeffs = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.25, 0.75],
        ]
    )
    graph_index = torch.tensor([0, 1, 0, 1])
    globals_ = MOLEGlobals(coefficients=coeffs, graph_index=graph_index)

    def fail_if_old_loop_is_used(*args, **kwargs):
        raise AssertionError("compact graph_index path should not use per-group Python F.linear loop")

    globals_.indexed_groups = fail_if_old_loop_is_used
    out = layer._forward_indexed_grouped(x, globals_)

    expected = []
    shared_w = layer.weight_shared.sum(0)
    shared_b = layer.bias_shared.sum(0)
    for x_edge, coeff_idx in zip(x, graph_index):
        c = coeffs[coeff_idx]
        routed_w = torch.einsum("e,eoi->oi", c, layer.weight_experts)
        routed_b = torch.einsum("e,eo->o", c, layer.bias_experts)
        expected.append(_linear_expected(x_edge, routed_w + shared_w, routed_b + shared_b))
    expected = torch.stack(expected, dim=0)

    torch.testing.assert_close(out, expected)


def test_mole_linear_edge_level_coefficients_backpropagate():
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLELinear

    layer = MOLELinear(2, 1, num_experts=3, num_shared_experts=0, bias=False)
    with torch.no_grad():
        layer.weight_experts.copy_(
            torch.tensor([[[1.0, 0.0]], [[0.0, 2.0]], [[-1.0, 1.0]]])
        )

    x = torch.tensor([[2.0, 4.0], [1.0, 3.0]], requires_grad=True)
    coeffs = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.25, 0.75]],
        requires_grad=True,
    )

    out = layer(x, MOLEGlobals(coefficients=coeffs))
    out.square().sum().backward()

    assert x.grad is not None
    assert layer.weight_experts.grad is not None
    assert coeffs.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.weight_experts.grad).all()
    assert torch.isfinite(coeffs.grad).all()
    assert coeffs.grad[0, 0].abs() > 0
    assert coeffs.grad[1, 1].abs() > 0
    assert coeffs.grad[1, 2].abs() > 0


def test_edge_moe_unique_type_routing_builds_compact_graph_index():
    import torch.nn as nn

    from dptb.nn.embedding.lem_moe_v3_edge import LemMoEV3Edge
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals

    class CaptureRouter:
        num_experts = 2

        def __call__(self, inputs, sizes=None):
            self.inputs = inputs
            self.sizes = sizes
            coeffs = inputs.new_tensor([[1.0, 0.0], [0.25, 0.75]])
            return coeffs, inputs.new_tensor(0.5), inputs.new_tensor(0.0)

    model = LemMoEV3Edge.__new__(LemMoEV3Edge)
    nn.Module.__init__(model)
    model.num_experts = 2
    model.edge_router_in_features = 2
    model.edge_router_unique_types = True
    model.edge_moe_compact_dispatch = True
    model.edge_moe_compact_min_edges = 0
    model.router = CaptureRouter()

    active_edge_one_hot = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    active_bond_type = torch.tensor([3, 3, 7])

    globals_, monitor_val, expert_load_cv, num_route_tokens = model._make_edge_moe_globals(
        active_edge_one_hot,
        active_bond_type,
    )

    assert isinstance(globals_, MOLEGlobals)
    assert globals_.sizes is None
    torch.testing.assert_close(model.router.inputs, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    torch.testing.assert_close(model.router.sizes, torch.tensor([2.0, 1.0]))
    torch.testing.assert_close(globals_.coefficients, torch.tensor([[1.0, 0.0], [0.25, 0.75]]))
    assert globals_.graph_index.tolist() == [0, 0, 1]
    assert float(monitor_val) == 0.5
    assert float(expert_load_cv) == 0.0
    assert float(num_route_tokens) == 2.0


def test_edge_moe_compact_dispatch_keeps_topk_unique_types_compact():
    import torch.nn as nn

    from dptb.nn.embedding.lem_moe_v3_edge import LemMoEV3Edge

    class CaptureRouter:
        num_experts = 2
        top_k = 1

        def __call__(self, inputs, sizes=None):
            coeffs = inputs.new_tensor([[1.0, 0.0], [0.0, 1.0]])
            return coeffs, inputs.new_tensor(0.5), inputs.new_tensor(0.0)

    model = LemMoEV3Edge.__new__(LemMoEV3Edge)
    nn.Module.__init__(model)
    model.num_experts = 2
    model.edge_router_in_features = 2
    model.edge_router_unique_types = True
    model.edge_moe_compact_dispatch = True
    model.edge_moe_compact_min_edges = 0
    model.router = CaptureRouter()

    active_edge_one_hot = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    active_bond_type = torch.tensor([3, 3, 7])

    globals_, _, _, num_route_tokens = model._make_edge_moe_globals(
        active_edge_one_hot,
        active_bond_type,
    )

    assert globals_.graph_index.tolist() == [0, 0, 1]
    torch.testing.assert_close(globals_.coefficients, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    assert float(num_route_tokens) == 2.0


def test_edge_moe_embedding_registers_separately_from_global_moe():
    from dptb.nn.embedding.emb import Embedding
    from dptb.nn.embedding.lem_moe_v3 import LemMoEV3
    from dptb.nn.embedding.lem_moe_v3_edge import LemMoEV3Edge, LemMoEV3EdgeH0

    assert Embedding._register["lem_moe_v3"] is LemMoEV3
    assert Embedding._register["lem_moe_v3_edge"] is LemMoEV3Edge
    assert Embedding._register["lem_moe_v3_edge_h0"] is LemMoEV3EdgeH0
