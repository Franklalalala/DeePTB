import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F


def _split_loop_linear(x, mixed_weights, mixed_bias, split_sizes):
    parts = []
    start = 0
    for graph_id, rows in enumerate(split_sizes):
        rows = int(rows)
        bias = mixed_bias[graph_id] if mixed_bias is not None else None
        parts.append(F.linear(x[start:start + rows], mixed_weights[graph_id], bias))
        start += rows
    return torch.cat(parts, dim=0)


@pytest.mark.parametrize("x_rank", [2, 3])
@pytest.mark.parametrize("with_bias", [False, True])
def test_grouped_linear_apply_matches_split_loop_cpu(x_rank, with_bias):
    from dptb.nn import so2_triton_grouped_linear_ops as ops

    torch.manual_seed(20260423)
    dtype = torch.float64
    split_sizes = (3, 5, 2, 4)
    n_rows = sum(split_sizes)
    in_features = 7 if x_rank == 3 else 11
    out_features = 9 if x_rank == 3 else 13
    x_shape = (n_rows, 2, in_features) if x_rank == 3 else (n_rows, in_features)

    x = torch.randn(*x_shape, dtype=dtype, requires_grad=True)
    mixed_weights = torch.randn(len(split_sizes), out_features, in_features, dtype=dtype, requires_grad=True)
    mixed_bias = (
        torch.randn(len(split_sizes), out_features, dtype=dtype, requires_grad=True)
        if with_bias else None
    )

    y_ref = _split_loop_linear(x, mixed_weights, mixed_bias, split_sizes)
    y_new = ops.grouped_linear_apply(x, mixed_weights, mixed_bias, split_sizes)
    torch.testing.assert_close(y_new, y_ref, atol=1e-10, rtol=1e-10)

    y_ref.square().sum().backward()
    gx_ref = x.grad.detach().clone()
    gw_ref = mixed_weights.grad.detach().clone()
    gb_ref = mixed_bias.grad.detach().clone() if mixed_bias is not None else None

    x.grad.zero_()
    mixed_weights.grad.zero_()
    if mixed_bias is not None:
        mixed_bias.grad.zero_()
    y_new.square().sum().backward()

    torch.testing.assert_close(x.grad, gx_ref, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(mixed_weights.grad, gw_ref, atol=1e-10, rtol=1e-10)
    if mixed_bias is not None:
        torch.testing.assert_close(mixed_bias.grad, gb_ref, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("x_rank", [2, 3])
@pytest.mark.parametrize("num_shared_experts", [0, 2])
def test_mole_linear_triton_grouped_matches_split_loop_cpu(x_rank, num_shared_experts):
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLELinear

    torch.manual_seed(20260423)
    dtype = torch.float64
    split_sizes = (3, 5, 2, 4)
    n_rows = sum(split_sizes)
    num_experts = 6
    in_features = 7 if x_rank == 3 else 11
    out_features = 9 if x_rank == 3 else 13
    x_shape = (n_rows, 2, in_features) if x_rank == 3 else (n_rows, in_features)

    coeffs = torch.rand(len(split_sizes), num_experts, dtype=dtype)
    coeffs = coeffs / coeffs.sum(dim=-1, keepdim=True)
    mole_globals = MOLEGlobals(coefficients=coeffs, split_sizes=split_sizes)

    base = MOLELinear(
        in_features,
        out_features,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        bias=True,
        mole_linear_mode="split_loop",
    ).to(dtype=dtype)
    triton = MOLELinear(
        in_features,
        out_features,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        bias=True,
        mole_linear_mode="triton_grouped_linear",
    ).to(dtype=dtype)
    triton.load_state_dict(base.state_dict(), strict=True)

    x0 = torch.randn(*x_shape, dtype=dtype, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)
    y0 = base(x0, mole_globals)
    y1 = triton(x1, mole_globals)
    torch.testing.assert_close(y1, y0, atol=1e-10, rtol=1e-10)

    probe = torch.randn_like(y0)
    (y0 * probe).sum().backward()
    (y1 * probe).sum().backward()
    torch.testing.assert_close(x1.grad, x0.grad, atol=1e-10, rtol=1e-10)

    for (name0, param0), (name1, param1) in zip(base.named_parameters(), triton.named_parameters()):
        assert name0 == name1
        if param0.grad is None:
            assert param1.grad is None
        else:
            torch.testing.assert_close(param1.grad, param0.grad, atol=1e-10, rtol=1e-10)


def test_grouped_linear_apply_cuda_fp32_if_available(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("Triton grouped linear CUDA smoke requires CUDA")

    from dptb.nn import so2_triton_grouped_linear_ops as ops

    monkeypatch.setenv("DPTB_TRITON_LINEAR_REQUIRE", "1")
    torch.manual_seed(20260423)
    device = torch.device("cuda")
    dtype = torch.float32
    split_sizes = (13, 17, 19, 11)
    n_rows = sum(split_sizes)
    in_features = 37
    out_features = 41

    x = torch.randn(n_rows, 2, in_features, device=device, dtype=dtype, requires_grad=True)
    mixed_weights = torch.randn(
        len(split_sizes), out_features, in_features, device=device, dtype=dtype, requires_grad=True
    )
    mixed_bias = torch.randn(len(split_sizes), out_features, device=device, dtype=dtype, requires_grad=True)

    y_ref = _split_loop_linear(x, mixed_weights, mixed_bias, split_sizes)
    y_new = ops.grouped_linear_apply(x, mixed_weights, mixed_bias, split_sizes)
    torch.testing.assert_close(y_new, y_ref, atol=1e-4, rtol=1e-4)

    y_ref.square().mean().backward()
    gx_ref = x.grad.detach().clone()
    gw_ref = mixed_weights.grad.detach().clone()
    gb_ref = mixed_bias.grad.detach().clone()

    x.grad.zero_()
    mixed_weights.grad.zero_()
    mixed_bias.grad.zero_()
    y_new.square().mean().backward()

    torch.testing.assert_close(x.grad, gx_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(mixed_weights.grad, gw_ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(mixed_bias.grad, gb_ref, atol=1e-4, rtol=1e-4)


def test_mole_linear_env_selects_triton_grouped(monkeypatch):
    from dptb.nn.tensor_product_moe_v3 import MOLELinear

    monkeypatch.setenv("DPTB_MOLE_LINEAR_MODE", "triton_grouped_linear")
    layer = MOLELinear(3, 5, num_experts=2, num_shared_experts=0)

    assert layer.mole_linear_mode == "triton_grouped_linear"


def test_grouped_linear_require_raises_when_backend_unavailable(monkeypatch):
    from dptb.nn import so2_triton_grouped_linear_ops as ops

    monkeypatch.setenv("DPTB_TRITON_LINEAR_REQUIRE", "1")
    x = torch.randn(4, 3)
    mixed_weights = torch.randn(2, 5, 3)
    split_sizes = (2, 2)

    with pytest.raises(RuntimeError, match="Triton grouped linear backend is unavailable"):
        ops.grouped_linear_apply(x, mixed_weights, None, split_sizes)
