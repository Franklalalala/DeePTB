import pytest
import sys
import types


@pytest.fixture(autouse=True)
def _clear_mole_linear_mode_env(monkeypatch):
    monkeypatch.delenv("DPTB_MOLE_LINEAR_MODE", raising=False)


def _make_globals(torch, *, device, dtype, sizes=(3, 5, 2, 7), num_experts=8):
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals

    sizes = torch.tensor(sizes, device=device)
    coeffs = torch.rand(int(sizes.numel()), num_experts, device=device, dtype=dtype)
    coeffs = coeffs / coeffs.sum(dim=-1, keepdim=True)
    return MOLEGlobals(coefficients=coeffs, sizes=sizes), int(sizes.sum().item())


def _assert_mole_modes_match(torch, *, shape, bias, num_shared_experts, device, dtype, sizes=(3, 5, 2, 7)):
    from dptb.nn.tensor_product_moe_v3 import MOLELinear

    num_experts = 8
    in_features = shape[-1]
    out_features = 13
    globals_, _ = _make_globals(torch, device=device, dtype=dtype, sizes=sizes, num_experts=num_experts)

    base = MOLELinear(
        in_features,
        out_features,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        bias=bias,
        mole_linear_mode="split_loop",
    ).to(device=device, dtype=dtype)
    indexed = MOLELinear(
        in_features,
        out_features,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        bias=bias,
        mole_linear_mode="indexed_ref",
    ).to(device=device, dtype=dtype)
    indexed.load_state_dict(base.state_dict(), strict=True)

    x0 = torch.randn(*shape, device=device, dtype=dtype, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)

    y0 = base(x0, globals_)
    y1 = indexed(x1, globals_)
    torch.testing.assert_close(y1, y0, atol=1e-10, rtol=1e-10)

    probe = torch.randn_like(y0)
    (y0 * probe).sum().backward()
    (y1 * probe).sum().backward()

    torch.testing.assert_close(x1.grad, x0.grad, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(
        indexed.weight_experts.grad,
        base.weight_experts.grad,
        atol=1e-10,
        rtol=1e-10,
    )
    if bias:
        torch.testing.assert_close(
            indexed.bias_experts.grad,
            base.bias_experts.grad,
            atol=1e-10,
            rtol=1e-10,
        )
    if num_shared_experts > 0:
        torch.testing.assert_close(
            indexed.weight_shared.grad,
            base.weight_shared.grad,
            atol=1e-10,
            rtol=1e-10,
        )
        if bias:
            torch.testing.assert_close(
                indexed.bias_shared.grad,
                base.bias_shared.grad,
                atol=1e-10,
                rtol=1e-10,
            )


def test_mole_linear_indexed_ref_matches_split_loop_forward_and_grad():
    torch = pytest.importorskip("torch")

    torch.manual_seed(20260423)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    _, n_edges = _make_globals(torch, device=device, dtype=dtype)

    _assert_mole_modes_match(
        torch,
        shape=(n_edges, 11),
        bias=True,
        num_shared_experts=2,
        device=device,
        dtype=dtype,
    )
    _assert_mole_modes_match(
        torch,
        shape=(n_edges, 2, 11),
        bias=True,
        num_shared_experts=2,
        device=device,
        dtype=dtype,
    )


def test_mole_linear_indexed_ref_matches_split_loop_without_bias_or_shared_experts():
    torch = pytest.importorskip("torch")

    torch.manual_seed(20260424)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    _, n_edges = _make_globals(torch, device=device, dtype=dtype, sizes=(1, 4, 6))

    _assert_mole_modes_match(
        torch,
        shape=(n_edges, 2, 7),
        bias=False,
        num_shared_experts=0,
        device=device,
        dtype=dtype,
        sizes=(1, 4, 6),
    )
    _assert_mole_modes_match(
        torch,
        shape=(n_edges, 7),
        bias=True,
        num_shared_experts=0,
        device=device,
        dtype=dtype,
        sizes=(1, 4, 6),
    )
    _assert_mole_modes_match(
        torch,
        shape=(n_edges, 2, 7),
        bias=False,
        num_shared_experts=2,
        device=device,
        dtype=dtype,
        sizes=(1, 4, 6),
    )


def test_expand_graph_index_cached_reuses_expanded_tensor():
    torch = pytest.importorskip("torch")
    from dptb.nn.tensor_product_moe_v3 import (
        MOLEGlobals,
        _expand_graph_index_cached,
        _expand_graph_index_for_leading_dims,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_index = torch.tensor([0, 0, 1, 1, 1], device=device, dtype=torch.long)
    x = torch.randn(graph_index.numel(), 3, 4, device=device)
    globals_ = MOLEGlobals(
        coefficients=torch.ones(2, 4, device=device),
        graph_index=graph_index,
    )

    expected = _expand_graph_index_for_leading_dims(graph_index, x)
    first = _expand_graph_index_cached(graph_index, x, globals_)
    second = _expand_graph_index_cached(graph_index, x, globals_)

    torch.testing.assert_close(first, expected)
    assert first is second

    x_other_shape = torch.randn(graph_index.numel(), 2, 4, device=device)
    third = _expand_graph_index_cached(graph_index, x_other_shape, globals_)
    assert third is not first

    x_2d = torch.randn(graph_index.numel(), 4, device=device)
    assert _expand_graph_index_cached(graph_index, x_2d, globals_) is graph_index


def test_mole_globals_caches_indexed_flat_permutation():
    torch = pytest.importorskip("torch")
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals

    graph_index = torch.tensor([2, 0, 2, 1, 0], dtype=torch.long)
    x = torch.randn(graph_index.numel(), 2, 4)
    globals_ = MOLEGlobals(
        coefficients=torch.ones(3, 4),
        graph_index=graph_index,
    )

    permute_idx, unpermute_idx, sorted_graph_index = globals_.indexed_flat_permutation(graph_index, x)
    cached = globals_.indexed_flat_permutation(graph_index, x)

    assert sorted_graph_index.tolist() == [0, 0, 0, 0, 1, 1, 2, 2, 2, 2]
    restored = torch.arange(sorted_graph_index.numel()).index_select(0, permute_idx).index_select(0, unpermute_idx)
    assert restored.tolist() == list(range(sorted_graph_index.numel()))
    assert cached[0] is permute_idx
    assert cached[1] is unpermute_idx
    assert cached[2] is sorted_graph_index

    sorted_globals = MOLEGlobals(
        coefficients=torch.ones(3, 4),
        graph_index=torch.tensor([0, 0, 1, 1, 2], dtype=torch.long),
    )
    sorted_x = torch.randn(5, 4)
    permute_idx, unpermute_idx, sorted_graph_index = sorted_globals.indexed_flat_permutation(
        sorted_globals.graph_index,
        sorted_x,
    )
    assert permute_idx is None
    assert unpermute_idx is None
    assert sorted_graph_index.tolist() == sorted_globals.graph_index.tolist()


def test_cueq_indexed_linear_sorts_graph_index_and_restores_order(monkeypatch):
    torch = pytest.importorskip("torch")
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLELinear

    layer = MOLELinear(
        2,
        2,
        num_experts=3,
        num_shared_experts=0,
        bias=False,
        mole_linear_mode="cueq_indexed_linear",
    )
    graph_index = torch.tensor([1, 0, 1, 2], dtype=torch.long)
    globals_ = MOLEGlobals(coefficients=torch.ones(3, 3), graph_index=graph_index)
    x = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )
    mixed_weights = torch.zeros(3, 2, 2)

    class FakeCueIndexedLinear:
        def __call__(self, flat_x, *, weight, weight_indices):
            assert torch.all(weight_indices[1:] >= weight_indices[:-1])
            return flat_x + weight_indices.to(flat_x.dtype).unsqueeze(1) * 100.0

    monkeypatch.setattr(layer, "_get_cueq_indexed_linear", lambda *args, **kwargs: FakeCueIndexedLinear())
    monkeypatch.setattr(layer, "_infer_cueq_weight_order", lambda *args, **kwargs: "io_scaled")
    monkeypatch.setattr(layer, "_cueq_flatten_weight", lambda weights, order: weights)

    out = layer._apply_cueq_indexed_linear(x, mixed_weights, None, graph_index, globals_)

    expected = x + graph_index.to(x.dtype).unsqueeze(1) * 100.0
    torch.testing.assert_close(out, expected)


def test_mole_linear_fallback_average_ignores_indexed_mode():
    torch = pytest.importorskip("torch")
    from dptb.nn.tensor_product_moe_v3 import MOLELinear

    torch.manual_seed(20260425)
    dtype = torch.float64
    base = MOLELinear(5, 3, num_experts=4, num_shared_experts=1, bias=True, mole_linear_mode="split_loop").to(dtype=dtype)
    indexed = MOLELinear(5, 3, num_experts=4, num_shared_experts=1, bias=True, mole_linear_mode="indexed_ref").to(dtype=dtype)
    indexed.load_state_dict(base.state_dict(), strict=True)

    x = torch.randn(6, 2, 5, dtype=dtype)
    torch.testing.assert_close(indexed(x, None), base(x, None), atol=1e-10, rtol=1e-10)


def test_mole_linear_indexed_ref_matches_coefficients_grad():
    torch = pytest.importorskip("torch")
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLELinear

    torch.manual_seed(20260426)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    sizes = (2, 3, 4)
    num_graphs = len(sizes)
    num_experts = 5
    n_edges = sum(sizes)

    coeffs = torch.rand(num_graphs, num_experts, device=device, dtype=dtype)
    coeffs = (coeffs / coeffs.sum(dim=-1, keepdim=True)).detach()
    coeffs0 = coeffs.clone().requires_grad_(True)
    coeffs1 = coeffs.clone().requires_grad_(True)
    globals0 = MOLEGlobals(coefficients=coeffs0, split_sizes=sizes)
    globals1 = MOLEGlobals(coefficients=coeffs1, split_sizes=sizes)

    base = MOLELinear(6, 8, num_experts=num_experts, num_shared_experts=1, bias=True, mole_linear_mode="split_loop").to(device=device, dtype=dtype)
    indexed = MOLELinear(6, 8, num_experts=num_experts, num_shared_experts=1, bias=True, mole_linear_mode="indexed_ref").to(device=device, dtype=dtype)
    indexed.load_state_dict(base.state_dict(), strict=True)

    x0 = torch.randn(n_edges, 2, 6, device=device, dtype=dtype, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)
    probe = torch.randn(n_edges, 2, 8, device=device, dtype=dtype)

    (base(x0, globals0) * probe).sum().backward()
    (indexed(x1, globals1) * probe).sum().backward()

    torch.testing.assert_close(coeffs1.grad, coeffs0.grad, atol=1e-10, rtol=1e-10)


def test_mole_globals_explicit_split_sizes_take_precedence():
    torch = pytest.importorskip("torch")
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLELinear

    torch.manual_seed(20260427)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    num_experts = 5
    split_sizes = (2, 4, 3)
    misleading_sizes = torch.tensor((1, 1, 1), device=device)

    coeffs = torch.rand(len(split_sizes), num_experts, device=device, dtype=dtype)
    coeffs = coeffs / coeffs.sum(dim=-1, keepdim=True)
    globals_ = MOLEGlobals(coefficients=coeffs, sizes=misleading_sizes, split_sizes=split_sizes)

    base = MOLELinear(6, 8, num_experts=num_experts, num_shared_experts=1, bias=True, mole_linear_mode="split_loop").to(device=device, dtype=dtype)
    indexed = MOLELinear(6, 8, num_experts=num_experts, num_shared_experts=1, bias=True, mole_linear_mode="indexed_ref").to(device=device, dtype=dtype)
    indexed.load_state_dict(base.state_dict(), strict=True)

    x = torch.randn(sum(split_sizes), 6, device=device, dtype=dtype)
    torch.testing.assert_close(indexed(x, globals_), base(x, globals_), atol=1e-10, rtol=1e-10)


def test_mole_globals_explicit_split_sizes_override_graph_index_for_indexed_modes():
    torch = pytest.importorskip("torch")
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLELinear, _mole_graph_index

    torch.manual_seed(20260428)
    dtype = torch.float64
    num_experts = 5
    split_sizes = (2, 1)
    misleading_graph_index = torch.tensor([1, 0, 0], dtype=torch.long)

    coeffs = torch.rand(len(split_sizes), num_experts, dtype=dtype)
    coeffs = coeffs / coeffs.sum(dim=-1, keepdim=True)
    globals_ = MOLEGlobals(
        coefficients=coeffs,
        split_sizes=split_sizes,
        graph_index=misleading_graph_index,
    )

    resolved = _mole_graph_index(globals_, sum(split_sizes), device=torch.device("cpu"))
    torch.testing.assert_close(resolved, torch.tensor([0, 0, 1], dtype=torch.long))

    base = MOLELinear(
        4,
        3,
        num_experts=num_experts,
        num_shared_experts=0,
        bias=True,
        mole_linear_mode="split_loop",
    ).to(dtype=dtype)
    indexed = MOLELinear(
        4,
        3,
        num_experts=num_experts,
        num_shared_experts=0,
        bias=True,
        mole_linear_mode="indexed_ref",
    ).to(dtype=dtype)
    indexed.load_state_dict(base.state_dict(), strict=True)

    x = torch.randn(sum(split_sizes), 4, dtype=dtype)
    torch.testing.assert_close(indexed(x, globals_), base(x, globals_), atol=1e-10, rtol=1e-10)


def test_mole_globals_cuda_graph_index_avoids_split_tuple_sync():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA graph-index fast path requires CUDA")

    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, _mole_graph_index

    sizes = torch.tensor((2, 4, 3), device="cuda")
    graph_index = torch.repeat_interleave(
        torch.arange(sizes.numel(), device="cuda", dtype=torch.long), sizes
    )
    coeffs = torch.rand(sizes.numel(), 5, device="cuda")
    globals_ = MOLEGlobals(coefficients=coeffs, sizes=sizes, graph_index=graph_index)

    assert globals_.split_sizes is None
    resolved = _mole_graph_index(globals_, int(sizes.sum().item()), device=torch.device("cuda"))
    assert resolved.data_ptr() == graph_index.data_ptr()


def test_mole_linear_env_selects_indexed_ref(monkeypatch):
    torch = pytest.importorskip("torch")
    from dptb.nn.tensor_product_moe_v3 import MOLELinear

    monkeypatch.setenv("DPTB_MOLE_LINEAR_MODE", "indexed_ref")
    layer = MOLELinear(4, 4)
    assert layer.mole_linear_mode == "indexed_ref"


def test_mole_linear_topk_mixing_matches_dense_coefficients():
    torch = pytest.importorskip("torch")
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLELinear

    torch.manual_seed(20260521)
    layer = MOLELinear(
        4,
        3,
        num_experts=5,
        num_shared_experts=1,
        bias=True,
        mole_linear_mode="indexed_ref",
    ).to(dtype=torch.float64)
    topk_indices = torch.tensor([[0, 2], [3, 1], [4, 0]], dtype=torch.long)
    topk_values = torch.tensor([[0.25, 0.75], [0.6, 0.4], [0.9, 0.1]], dtype=torch.float64)
    coeffs = torch.zeros(3, 5, dtype=torch.float64)
    coeffs.scatter_(1, topk_indices, topk_values)

    dense_globals = MOLEGlobals(coefficients=coeffs)
    topk_globals = MOLEGlobals(
        coefficients=coeffs,
        topk_indices=topk_indices,
        topk_values=topk_values,
    )

    dense_w, dense_b = layer._mix_expert_parameters(dense_globals)
    topk_w, topk_b = layer._mix_expert_parameters(topk_globals)

    torch.testing.assert_close(topk_w, dense_w, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(topk_b, dense_b, atol=1e-12, rtol=1e-12)


def test_mole_linear_env_selects_cublas_grouped(monkeypatch):
    torch = pytest.importorskip("torch")
    from dptb.nn.tensor_product_moe_v3 import MOLELinear

    monkeypatch.setenv("DPTB_MOLE_LINEAR_MODE", "cublas_grouped")
    layer = MOLELinear(4, 4)
    assert layer.mole_linear_mode == "cublas_grouped"


def test_mole_linear_invalid_mode_rejected():
    pytest.importorskip("torch")
    from dptb.nn.tensor_product_moe_v3 import MOLELinear

    with pytest.raises(ValueError, match="mole_linear_mode"):
        MOLELinear(4, 4, mole_linear_mode="bad")


def test_mole_linear_cueq_single_graph_matches_split_loop_without_cueq_import():
    torch = pytest.importorskip("torch")
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLELinear

    torch.manual_seed(20260505)
    dtype = torch.float64
    n_rows = 7
    num_experts = 5
    in_features = 4
    out_features = 6

    coeffs = torch.rand(1, num_experts, dtype=dtype)
    coeffs = (coeffs / coeffs.sum(dim=-1, keepdim=True)).detach()
    coeffs0 = coeffs.clone().requires_grad_(True)
    coeffs1 = coeffs.clone().requires_grad_(True)
    globals0 = MOLEGlobals(coefficients=coeffs0, split_sizes=(n_rows,))
    globals1 = MOLEGlobals(coefficients=coeffs1, split_sizes=(n_rows,))

    base = MOLELinear(
        in_features,
        out_features,
        num_experts=num_experts,
        num_shared_experts=1,
        bias=True,
        mole_linear_mode="split_loop",
    ).to(dtype=dtype)
    cueq = MOLELinear(
        in_features,
        out_features,
        num_experts=num_experts,
        num_shared_experts=1,
        bias=True,
        mole_linear_mode="cueq_indexed_linear",
    ).to(dtype=dtype)
    cueq.load_state_dict(base.state_dict(), strict=True)

    x0 = torch.randn(n_rows, 2, in_features, dtype=dtype, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)
    y0 = base(x0, globals0)
    y1 = cueq(x1, globals1)
    torch.testing.assert_close(y1, y0, atol=1e-10, rtol=1e-10)

    probe = torch.randn_like(y0)
    (y0 * probe).sum().backward()
    (y1 * probe).sum().backward()

    torch.testing.assert_close(x1.grad, x0.grad, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(coeffs1.grad, coeffs0.grad, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(cueq.weight_experts.grad, base.weight_experts.grad, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(cueq.bias_experts.grad, base.bias_experts.grad, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(cueq.weight_shared.grad, base.weight_shared.grad, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(cueq.bias_shared.grad, base.bias_shared.grad, atol=1e-10, rtol=1e-10)
    assert cueq._cueq_indexed_linear_cache == {}


def test_mole_linear_cueq_indexed_smoke_if_available():
    torch = pytest.importorskip("torch")
    pytest.importorskip("cuequivariance")
    pytest.importorskip("cuequivariance_torch")
    if not torch.cuda.is_available():
        pytest.skip("cueq indexed linear smoke requires CUDA")

    from dptb.nn.tensor_product_moe_v3 import MOLELinear

    torch.manual_seed(20260423)
    device = torch.device("cuda")
    dtype = torch.float32
    sizes = (4, 6, 5)
    num_experts = 6
    in_features = 7
    out_features = 9
    globals_, n_edges = _make_globals(
        torch,
        device=device,
        dtype=dtype,
        sizes=sizes,
        num_experts=num_experts,
    )

    base = MOLELinear(
        in_features,
        out_features,
        num_experts=num_experts,
        num_shared_experts=1,
        bias=True,
        mole_linear_mode="split_loop",
    ).to(device=device, dtype=dtype)
    cueq = MOLELinear(
        in_features,
        out_features,
        num_experts=num_experts,
        num_shared_experts=1,
        bias=True,
        mole_linear_mode="cueq_indexed_linear",
    ).to(device=device, dtype=dtype)
    cueq.load_state_dict(base.state_dict(), strict=True)

    x0 = torch.randn(n_edges, 2, in_features, device=device, dtype=dtype, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)
    y0 = base(x0, globals_)
    y1 = cueq(x1, globals_)
    torch.testing.assert_close(y1, y0, atol=2e-4, rtol=2e-4)

    probe = torch.randn_like(y0)
    (y0 * probe).mean().backward()
    (y1 * probe).mean().backward()
    torch.testing.assert_close(x1.grad, x0.grad, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(cueq.weight_experts.grad, base.weight_experts.grad, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(cueq.bias_experts.grad, base.bias_experts.grad, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(cueq.weight_shared.grad, base.weight_shared.grad, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(cueq.bias_shared.grad, base.bias_shared.grad, atol=2e-4, rtol=2e-4)
    assert cueq._cueq_weight_order == "io_scaled"


def test_mole_linear_cublas_grouped_smoke_if_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("cuBLAS grouped GEMM smoke requires CUDA")

    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLELinear

    torch.manual_seed(20260521)
    device = torch.device("cuda")
    dtype = torch.float32
    split_sizes = (5, 0, 7, 4)
    num_graphs = len(split_sizes)
    num_experts = 6
    in_features = 5
    out_features = 8
    n_edges = sum(split_sizes)
    coeffs = torch.rand(num_graphs, num_experts, device=device, dtype=dtype)
    coeffs = (coeffs / coeffs.sum(dim=-1, keepdim=True)).detach()
    coeffs0 = coeffs.clone().requires_grad_(True)
    coeffs1 = coeffs.clone().requires_grad_(True)
    globals0 = MOLEGlobals(coefficients=coeffs0, split_sizes=split_sizes)
    globals1 = MOLEGlobals(coefficients=coeffs1, split_sizes=split_sizes)

    base = MOLELinear(
        in_features,
        out_features,
        num_experts=num_experts,
        num_shared_experts=1,
        bias=True,
        mole_linear_mode="split_loop",
    ).to(device=device, dtype=dtype)
    cublas = MOLELinear(
        in_features,
        out_features,
        num_experts=num_experts,
        num_shared_experts=1,
        bias=True,
        mole_linear_mode="cublas_grouped",
    ).to(device=device, dtype=dtype)
    cublas.load_state_dict(base.state_dict(), strict=True)

    x0 = torch.randn(n_edges, 2, in_features, device=device, dtype=dtype, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)
    y0 = base(x0, globals0)
    y1 = cublas(x1, globals1)
    torch.testing.assert_close(y1, y0, atol=2e-4, rtol=2e-4)

    probe = torch.randn_like(y0)
    (y0 * probe).mean().backward()
    (y1 * probe).mean().backward()
    torch.testing.assert_close(x1.grad, x0.grad, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(coeffs1.grad, coeffs0.grad, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(cublas.weight_experts.grad, base.weight_experts.grad, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(cublas.bias_experts.grad, base.bias_experts.grad, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(cublas.weight_shared.grad, base.weight_shared.grad, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(cublas.bias_shared.grad, base.bias_shared.grad, atol=2e-4, rtol=2e-4)


def test_cublas_grouped_multi_smoke_if_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("cuBLAS grouped GEMM smoke requires CUDA")

    import torch.nn.functional as F
    from dptb.nn.cublas_grouped_gemm import grouped_gemm_multi

    torch.manual_seed(20260521)
    device = torch.device("cuda")
    ptr = torch.tensor([0, 4, 4, 9], dtype=torch.long)
    xs0 = [
        torch.randn(9, 5, device=device, requires_grad=True),
        torch.randn(9, 3, device=device, requires_grad=True),
    ]
    ws0 = [
        torch.randn(3, 7, 5, device=device, requires_grad=True),
        torch.randn(3, 4, 3, device=device, requires_grad=True),
    ]
    xs1 = [x.detach().clone().requires_grad_(True) for x in xs0]
    ws1 = [w.detach().clone().requires_grad_(True) for w in ws0]

    def ref_grouped(x, weight):
        parts = []
        for group in range(weight.shape[0]):
            start = int(ptr[group])
            end = int(ptr[group + 1])
            if end > start:
                parts.append(F.linear(x[start:end], weight[group]))
        return torch.cat(parts, dim=0)

    ref = [ref_grouped(x, w) for x, w in zip(xs0, ws0)]
    got = grouped_gemm_multi(xs1, [ptr, ptr], ws1)
    for got_i, ref_i in zip(got, ref):
        torch.testing.assert_close(got_i, ref_i, atol=2e-4, rtol=2e-4)

    probes = [torch.randn_like(y) for y in ref]
    sum((y * p).mean() for y, p in zip(ref, probes)).backward()
    sum((y * p).mean() for y, p in zip(got, probes)).backward()
    for x1, x0 in zip(xs1, xs0):
        torch.testing.assert_close(x1.grad, x0.grad, atol=2e-4, rtol=2e-4)
    for w1, w0 in zip(ws1, ws0):
        torch.testing.assert_close(w1.grad, w0.grad, atol=2e-4, rtol=2e-4)


def test_mole_linear_cueq_env_smoke_if_available(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("cuequivariance")
    pytest.importorskip("cuequivariance_torch")
    if not torch.cuda.is_available():
        pytest.skip("cueq indexed linear smoke requires CUDA")

    from dptb.nn.tensor_product_moe_v3 import MOLELinear

    monkeypatch.setenv("DPTB_MOLE_LINEAR_MODE", "cueq_indexed_linear")
    assert MOLELinear(3, 3).mole_linear_mode == "cueq_indexed_linear"


def test_mole_linear_cueq_cache_key_tracks_num_graphs(monkeypatch):
    torch = pytest.importorskip("torch")
    from dptb.nn.tensor_product_moe_v3 import MOLELinear
    from dptb.utils import cuda_cache_memory as probe

    constructed = []

    class FakeIrreps:
        def __init__(self, group, spec):
            self.group = group
            self.spec = spec

    class FakeLinear:
        def __init__(self, *args, **kwargs):
            constructed.append(kwargs["weight_classes"])

    monkeypatch.setitem(
        sys.modules,
        "cuequivariance",
        types.SimpleNamespace(O3=object(), Irreps=FakeIrreps, ir_mul=object()),
    )
    monkeypatch.setitem(
        sys.modules,
        "cuequivariance_torch",
        types.SimpleNamespace(Linear=FakeLinear),
    )
    probe.reset_cuda_cache_event_stats()
    probe.configure_cuda_cache_memory_monitor(enabled=False, event_enabled=True)

    layer = MOLELinear(4, 5, mole_linear_mode="cueq_indexed_linear")
    first = layer._get_cueq_indexed_linear(16, dtype=torch.float32, device=torch.device("cuda:0"))
    second = layer._get_cueq_indexed_linear(16, dtype=torch.float32, device=torch.device("cuda:0"))
    third = layer._get_cueq_indexed_linear(15, dtype=torch.float32, device=torch.device("cuda:0"))

    assert first is second
    assert third is not first
    assert constructed == [16, 15]
    stats = probe.cuda_cache_event_stats_snapshot()
    key16 = "cueq_indexed_linear|num_graphs=16|dtype=torch.float32|device=cuda:0|in_features=4|out_features=5"
    key15 = "cueq_indexed_linear|num_graphs=15|dtype=torch.float32|device=cuda:0|in_features=4|out_features=5"
    assert stats[key16]["hits"] == 1
    assert stats[key16]["misses"] == 1
    assert stats[key15]["misses"] == 1

    probe.reset_cuda_cache_event_stats()
    probe.configure_cuda_cache_memory_monitor(enabled=False, event_enabled=False)


def test_mole_linear_cueq_rejects_amp_dtype_if_available():
    torch = pytest.importorskip("torch")
    pytest.importorskip("cuequivariance")
    pytest.importorskip("cuequivariance_torch")
    if not torch.cuda.is_available():
        pytest.skip("cueq indexed linear smoke requires CUDA")

    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLELinear

    device = torch.device("cuda")
    dtype = torch.float16
    coeffs = torch.rand(2, 4, device=device, dtype=dtype)
    coeffs = coeffs / coeffs.sum(dim=-1, keepdim=True)
    globals_ = MOLEGlobals(coefficients=coeffs, split_sizes=(2, 3))
    layer = MOLELinear(
        3,
        5,
        num_experts=4,
        num_shared_experts=0,
        bias=False,
        mole_linear_mode="cueq_indexed_linear",
    ).to(device=device, dtype=dtype)
    x = torch.randn(5, 3, device=device, dtype=dtype)

    with pytest.raises(RuntimeError, match="float32/float64"):
        layer(x, globals_)
