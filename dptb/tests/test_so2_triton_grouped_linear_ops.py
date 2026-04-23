import os

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


def _split_loop_complex_linear(x_pair, mixed_weights, split_sizes):
    out_parts = []
    start = 0
    cout = mixed_weights.shape[1] // 2
    for graph_id, rows in enumerate(split_sizes):
        rows = int(rows)
        xr = x_pair[start:start + rows, 0, :]
        xi = x_pair[start:start + rows, 1, :]
        wr = mixed_weights[graph_id, :cout, :]
        wi = mixed_weights[graph_id, cout:, :]
        yr = xr.matmul(wr.transpose(0, 1)) - xi.matmul(wi.transpose(0, 1))
        yi = xr.matmul(wi.transpose(0, 1)) + xi.matmul(wr.transpose(0, 1))
        out_parts.append(torch.stack((yr, yi), dim=1))
        start += rows
    return torch.cat(out_parts, dim=0)


@pytest.mark.parametrize("x_rank", [2, 3])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("with_shared", [False, True])
@pytest.mark.parametrize("op_name", ["grouped_moe_fused_linear", "grouped_exact_moe_linear"])
def test_grouped_moe_linear_matches_materialized_cpu(x_rank, with_bias, with_shared, op_name):
    from dptb.nn import so2_triton_grouped_linear_ops as ops

    if os.environ.get("DPTB_TRITON_LINEAR_REQUIRE") == "1":
        pytest.skip("CPU parity uses torch fallback; CUDA tests cover required Triton execution")
    if op_name == "grouped_moe_fused_linear" and os.environ.get("DPTB_TRITON_LINEAR_REQUIRE") == "1":
        pytest.skip("row-tile fused expert Triton path is disabled unless explicitly enabled")

    torch.manual_seed(20260423)
    dtype = torch.float64
    split_sizes = (3, 5, 2, 4)
    n_rows = sum(split_sizes)
    num_experts = 6
    in_features = 7 if x_rank == 3 else 11
    out_features = 9 if x_rank == 3 else 13
    x_shape = (n_rows, 2, in_features) if x_rank == 3 else (n_rows, in_features)

    x0 = torch.randn(*x_shape, dtype=dtype, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)
    c0 = torch.rand(len(split_sizes), num_experts, dtype=dtype, requires_grad=True)
    c1 = c0.detach().clone().requires_grad_(True)
    w0 = torch.randn(num_experts, out_features, in_features, dtype=dtype, requires_grad=True)
    w1 = w0.detach().clone().requires_grad_(True)
    b0 = torch.randn(num_experts, out_features, dtype=dtype, requires_grad=True) if with_bias else None
    b1 = b0.detach().clone().requires_grad_(True) if b0 is not None else None
    sw0 = torch.randn(out_features, in_features, dtype=dtype, requires_grad=True) if with_shared else None
    sw1 = sw0.detach().clone().requires_grad_(True) if sw0 is not None else None
    sb0 = torch.randn(out_features, dtype=dtype, requires_grad=True) if (with_bias and with_shared) else None
    sb1 = sb0.detach().clone().requires_grad_(True) if sb0 is not None else None

    mixed = torch.einsum("ge,eoi->goi", c0, w0)
    if sw0 is not None:
        mixed = mixed + sw0.unsqueeze(0)
    mixed_bias = None
    if b0 is not None:
        mixed_bias = torch.einsum("ge,eo->go", c0, b0)
        if sb0 is not None:
            mixed_bias = mixed_bias + sb0.unsqueeze(0)

    y_ref = _split_loop_linear(x0, mixed, mixed_bias, split_sizes)
    y_new = getattr(ops, op_name)(x1, c1, w1, b1, sw1, sb1, split_sizes)
    torch.testing.assert_close(y_new, y_ref, atol=1e-10, rtol=1e-10)

    probe = torch.randn_like(y_ref)
    (y_ref * probe).sum().backward()
    (y_new * probe).sum().backward()

    torch.testing.assert_close(x1.grad, x0.grad, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(c1.grad, c0.grad, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(w1.grad, w0.grad, atol=1e-10, rtol=1e-10)
    if b0 is not None:
        torch.testing.assert_close(b1.grad, b0.grad, atol=1e-10, rtol=1e-10)
    if sw0 is not None:
        torch.testing.assert_close(sw1.grad, sw0.grad, atol=1e-10, rtol=1e-10)
    if sb0 is not None:
        torch.testing.assert_close(sb1.grad, sb0.grad, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("x_rank", [2, 3])
@pytest.mark.parametrize("with_bias", [False, True])
def test_grouped_linear_apply_matches_split_loop_cpu(x_rank, with_bias):
    from dptb.nn import so2_triton_grouped_linear_ops as ops

    if os.environ.get("DPTB_TRITON_LINEAR_REQUIRE") == "1":
        pytest.skip("CPU parity uses torch fallback; CUDA tests cover required Triton execution")

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


def test_grouped_exact_moe_linear_does_not_save_mixed_weights():
    from dptb.nn import so2_triton_grouped_linear_ops as ops

    if os.environ.get("DPTB_TRITON_LINEAR_REQUIRE") == "1":
        pytest.skip("saved tensor hook guard uses CPU fallback; CUDA tests cover required Triton execution")

    torch.manual_seed(20260424)
    dtype = torch.float64
    split_sizes = (3, 5, 2, 4)
    num_graphs = len(split_sizes)
    num_experts = 7
    in_features = 11
    out_features = 13

    x = torch.randn(sum(split_sizes), in_features, dtype=dtype, requires_grad=True)
    coeffs = torch.rand(num_graphs, num_experts, dtype=dtype, requires_grad=True)
    weights = torch.randn(num_experts, out_features, in_features, dtype=dtype, requires_grad=True)
    bias = torch.randn(num_experts, out_features, dtype=dtype, requires_grad=True)
    saved_shapes = []

    def _pack(tensor):
        saved_shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(_pack, lambda tensor: tensor):
        y = ops.grouped_exact_moe_linear(x, coeffs, weights, bias, None, None, split_sizes)
        y.square().sum().backward()

    assert (num_graphs, out_features, in_features) not in saved_shapes
    assert (num_graphs, out_features) not in saved_shapes


def test_grouped_complex_linear_matches_split_loop_cpu():
    from dptb.nn import so2_triton_grouped_linear_ops as ops

    if os.environ.get("DPTB_TRITON_LINEAR_REQUIRE") == "1":
        pytest.skip("CPU parity uses torch fallback; CUDA tests cover required Triton execution")

    torch.manual_seed(20260423)
    dtype = torch.float64
    split_sizes = (3, 5, 2, 4)
    n_rows = sum(split_sizes)
    in_features = 7
    out_features = 9

    x0 = torch.randn(n_rows, 2, in_features, dtype=dtype, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)
    w0 = torch.randn(len(split_sizes), 2 * out_features, in_features, dtype=dtype, requires_grad=True)
    w1 = w0.detach().clone().requires_grad_(True)

    y_ref = _split_loop_complex_linear(x0, w0, split_sizes)
    y_new = ops.grouped_complex_linear(x1, w1, split_sizes)
    torch.testing.assert_close(y_new, y_ref, atol=1e-10, rtol=1e-10)

    probe = torch.randn_like(y_ref)
    (y_ref * probe).sum().backward()
    (y_new * probe).sum().backward()

    torch.testing.assert_close(x1.grad, x0.grad, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(w1.grad, w0.grad, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("op_name", ["grouped_complex_moe_fused_linear", "grouped_complex_exact_moe_linear"])
@pytest.mark.parametrize("with_shared", [False, True])
def test_grouped_complex_moe_linear_matches_materialized_complex_cpu(op_name, with_shared):
    from dptb.nn import so2_triton_grouped_linear_ops as ops

    if os.environ.get("DPTB_TRITON_LINEAR_REQUIRE") == "1":
        pytest.skip("CPU parity uses torch fallback; CUDA tests cover required Triton execution")
    if op_name == "grouped_complex_moe_fused_linear" and os.environ.get("DPTB_TRITON_LINEAR_REQUIRE") == "1":
        pytest.skip("row-tile fused complex expert Triton path is disabled unless explicitly enabled")

    torch.manual_seed(20260423)
    dtype = torch.float64
    split_sizes = (3, 5, 2, 4)
    n_rows = sum(split_sizes)
    num_experts = 6
    in_features = 7
    out_features = 9

    x0 = torch.randn(n_rows, 2, in_features, dtype=dtype, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)
    c0 = torch.rand(len(split_sizes), num_experts, dtype=dtype, requires_grad=True)
    c1 = c0.detach().clone().requires_grad_(True)
    w0 = torch.randn(num_experts, 2 * out_features, in_features, dtype=dtype, requires_grad=True)
    w1 = w0.detach().clone().requires_grad_(True)
    sw0 = torch.randn(2 * out_features, in_features, dtype=dtype, requires_grad=True) if with_shared else None
    sw1 = sw0.detach().clone().requires_grad_(True) if sw0 is not None else None

    mixed = torch.einsum("ge,eoi->goi", c0, w0)
    if sw0 is not None:
        mixed = mixed + sw0.unsqueeze(0)
    y_ref = _split_loop_complex_linear(x0, mixed, split_sizes)
    if op_name == "grouped_complex_moe_fused_linear":
        if with_shared:
            pytest.skip("row-tile fused complex MoE path does not support shared weights")
        y_new = ops.grouped_complex_moe_fused_linear(x1, c1, w1, split_sizes)
    else:
        y_new = ops.grouped_complex_exact_moe_linear(x1, c1, w1, sw1, split_sizes)
    torch.testing.assert_close(y_new, y_ref, atol=1e-10, rtol=1e-10)

    probe = torch.randn_like(y_ref)
    (y_ref * probe).sum().backward()
    (y_new * probe).sum().backward()

    torch.testing.assert_close(x1.grad, x0.grad, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(c1.grad, c0.grad, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(w1.grad, w0.grad, atol=1e-10, rtol=1e-10)
    if sw0 is not None:
        torch.testing.assert_close(sw1.grad, sw0.grad, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("x_rank", [2, 3])
@pytest.mark.parametrize("num_shared_experts", [0, 2])
@pytest.mark.parametrize("mole_linear_mode", ["triton_grouped_linear", "triton_exact_grouped_linear"])
def test_mole_linear_triton_grouped_matches_split_loop_cpu(x_rank, num_shared_experts, mole_linear_mode):
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLELinear

    if os.environ.get("DPTB_TRITON_LINEAR_REQUIRE") == "1":
        pytest.skip("CPU parity uses torch fallback; CUDA tests cover required Triton execution")

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
        mole_linear_mode=mole_linear_mode,
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


@pytest.mark.parametrize("x_rank", [2, 3])
@pytest.mark.parametrize("num_shared_experts", [0, 2])
def test_mole_linear_triton_fused_expert_matches_split_loop_cpu(x_rank, num_shared_experts):
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLELinear

    if os.environ.get("DPTB_TRITON_LINEAR_REQUIRE") == "1":
        pytest.skip("CPU parity uses torch fallback; CUDA tests cover required Triton execution")

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
        mole_linear_mode="triton_fused_expert_linear",
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


def test_grouped_exact_moe_linear_cuda_fp32_if_available(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("Triton exact grouped MoE CUDA smoke requires CUDA")
    pytest.importorskip("triton")

    from dptb.nn import so2_triton_grouped_linear_ops as ops

    monkeypatch.setenv("DPTB_TRITON_LINEAR_REQUIRE", "1")
    torch.manual_seed(20260424)
    device = torch.device("cuda")
    dtype = torch.float32
    split_sizes = (13, 17, 19, 11)
    n_rows = sum(split_sizes)
    num_experts = 8
    in_features = 37
    out_features = 41

    x0 = torch.randn(n_rows, 2, in_features, device=device, dtype=dtype, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)
    c0 = torch.rand(len(split_sizes), num_experts, device=device, dtype=dtype, requires_grad=True)
    c1 = c0.detach().clone().requires_grad_(True)
    w0 = torch.randn(num_experts, out_features, in_features, device=device, dtype=dtype, requires_grad=True)
    w1 = w0.detach().clone().requires_grad_(True)
    b0 = torch.randn(num_experts, out_features, device=device, dtype=dtype, requires_grad=True)
    b1 = b0.detach().clone().requires_grad_(True)
    sw0 = torch.randn(out_features, in_features, device=device, dtype=dtype, requires_grad=True)
    sw1 = sw0.detach().clone().requires_grad_(True)
    sb0 = torch.randn(out_features, device=device, dtype=dtype, requires_grad=True)
    sb1 = sb0.detach().clone().requires_grad_(True)

    mixed = torch.einsum("ge,eoi->goi", c0, w0) + sw0.unsqueeze(0)
    mixed_bias = torch.einsum("ge,eo->go", c0, b0) + sb0.unsqueeze(0)
    y_ref = _split_loop_linear(x0, mixed, mixed_bias, split_sizes)
    y_new = ops.grouped_exact_moe_linear(x1, c1, w1, b1, sw1, sb1, split_sizes)
    torch.testing.assert_close(y_new, y_ref, atol=2e-4, rtol=2e-4)

    probe = torch.randn_like(y_ref)
    (y_ref * probe).sum().backward()
    (y_new * probe).sum().backward()

    torch.testing.assert_close(x1.grad, x0.grad, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(c1.grad, c0.grad, atol=3e-3, rtol=3e-4)
    torch.testing.assert_close(w1.grad, w0.grad, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(b1.grad, b0.grad, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(sw1.grad, sw0.grad, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(sb1.grad, sb0.grad, atol=3e-4, rtol=3e-4)


def test_grouped_moe_fused_linear_cuda_fp32_if_available(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("Triton grouped MoE CUDA smoke requires CUDA")
    if os.environ.get("DPTB_TRITON_LINEAR_ENABLE_FUSED_EXPERT", "0") != "1":
        pytest.skip("fused-expert Triton execution is disabled by default")
    pytest.importorskip("triton")

    from dptb.nn import so2_triton_grouped_linear_ops as ops

    monkeypatch.setenv("DPTB_TRITON_LINEAR_REQUIRE", "1")
    torch.manual_seed(20260423)
    device = torch.device("cuda")
    dtype = torch.float32
    split_sizes = (13, 17, 19, 11)
    n_rows = sum(split_sizes)
    num_experts = 8
    in_features = 37
    out_features = 41

    x0 = torch.randn(n_rows, 2, in_features, device=device, dtype=dtype, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)
    c0 = torch.rand(len(split_sizes), num_experts, device=device, dtype=dtype, requires_grad=True)
    c1 = c0.detach().clone().requires_grad_(True)
    w0 = torch.randn(num_experts, out_features, in_features, device=device, dtype=dtype, requires_grad=True)
    w1 = w0.detach().clone().requires_grad_(True)
    b0 = torch.randn(num_experts, out_features, device=device, dtype=dtype, requires_grad=True)
    b1 = b0.detach().clone().requires_grad_(True)
    sw0 = torch.randn(out_features, in_features, device=device, dtype=dtype, requires_grad=True)
    sw1 = sw0.detach().clone().requires_grad_(True)
    sb0 = torch.randn(out_features, device=device, dtype=dtype, requires_grad=True)
    sb1 = sb0.detach().clone().requires_grad_(True)

    mixed = torch.einsum("ge,eoi->goi", c0, w0) + sw0.unsqueeze(0)
    mixed_bias = torch.einsum("ge,eo->go", c0, b0) + sb0.unsqueeze(0)
    y_ref = _split_loop_linear(x0, mixed, mixed_bias, split_sizes)
    y_new = ops.grouped_moe_fused_linear(x1, c1, w1, b1, sw1, sb1, split_sizes)
    torch.testing.assert_close(y_new, y_ref, atol=2e-4, rtol=2e-4)

    probe = torch.randn_like(y_ref)
    (y_ref * probe).sum().backward()
    (y_new * probe).sum().backward()

    torch.testing.assert_close(x1.grad, x0.grad, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(c1.grad, c0.grad, atol=3e-3, rtol=3e-4)
    torch.testing.assert_close(w1.grad, w0.grad, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(b1.grad, b0.grad, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(sw1.grad, sw0.grad, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(sb1.grad, sb0.grad, atol=3e-4, rtol=3e-4)


def test_grouped_moe_fused_linear_require_raises_when_experimental_triton_disabled(monkeypatch):
    from dptb.nn import so2_triton_grouped_linear_ops as ops

    monkeypatch.setenv("DPTB_TRITON_LINEAR_REQUIRE", "1")
    monkeypatch.delenv("DPTB_TRITON_LINEAR_ENABLE_FUSED_EXPERT", raising=False)

    split_sizes = (2, 2)
    x = torch.randn(4, 3)
    coeffs = torch.rand(2, 5)
    weights = torch.randn(5, 7, 3)
    with pytest.raises(RuntimeError, match="fused-expert Triton execution is disabled"):
        ops.grouped_moe_fused_linear(x, coeffs, weights, None, None, None, split_sizes)


def test_grouped_complex_linear_cuda_fp32_if_available(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("Triton grouped complex linear CUDA smoke requires CUDA")
    pytest.importorskip("triton")

    from dptb.nn import so2_triton_grouped_linear_ops as ops

    monkeypatch.setenv("DPTB_TRITON_LINEAR_REQUIRE", "1")
    torch.manual_seed(20260423)
    device = torch.device("cuda")
    dtype = torch.float32
    split_sizes = (13, 17, 19, 11)
    n_rows = sum(split_sizes)
    in_features = 37
    out_features = 41

    x0 = torch.randn(n_rows, 2, in_features, device=device, dtype=dtype, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)
    w0 = torch.randn(
        len(split_sizes), 2 * out_features, in_features, device=device, dtype=dtype, requires_grad=True
    )
    w1 = w0.detach().clone().requires_grad_(True)

    y_ref = _split_loop_complex_linear(x0, w0, split_sizes)
    y_new = ops.grouped_complex_linear(x1, w1, split_sizes)
    torch.testing.assert_close(y_new, y_ref, atol=1e-4, rtol=1e-4)

    probe = torch.randn_like(y_ref)
    (y_ref * probe).sum().backward()
    (y_new * probe).sum().backward()

    torch.testing.assert_close(x1.grad, x0.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(w1.grad, w0.grad, atol=1e-4, rtol=1e-4)


def test_grouped_complex_exact_moe_linear_cuda_fp32_if_available(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("Triton complex exact grouped MoE CUDA smoke requires CUDA")
    pytest.importorskip("triton")

    from dptb.nn import so2_triton_grouped_linear_ops as ops

    monkeypatch.setenv("DPTB_TRITON_LINEAR_REQUIRE", "1")
    torch.manual_seed(20260424)
    device = torch.device("cuda")
    dtype = torch.float32
    split_sizes = (13, 17, 19, 11)
    n_rows = sum(split_sizes)
    num_experts = 8
    in_features = 37
    out_features = 41

    x0 = torch.randn(n_rows, 2, in_features, device=device, dtype=dtype, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)
    c0 = torch.rand(len(split_sizes), num_experts, device=device, dtype=dtype, requires_grad=True)
    c1 = c0.detach().clone().requires_grad_(True)
    w0 = torch.randn(num_experts, 2 * out_features, in_features, device=device, dtype=dtype, requires_grad=True)
    w1 = w0.detach().clone().requires_grad_(True)
    sw0 = torch.randn(2 * out_features, in_features, device=device, dtype=dtype, requires_grad=True)
    sw1 = sw0.detach().clone().requires_grad_(True)

    mixed = torch.einsum("ge,eoi->goi", c0, w0) + sw0.unsqueeze(0)
    y_ref = _split_loop_complex_linear(x0, mixed, split_sizes)
    y_new = ops.grouped_complex_exact_moe_linear(x1, c1, w1, sw1, split_sizes)
    torch.testing.assert_close(y_new, y_ref, atol=2e-4, rtol=2e-4)

    probe = torch.randn_like(y_ref)
    (y_ref * probe).sum().backward()
    (y_new * probe).sum().backward()

    torch.testing.assert_close(x1.grad, x0.grad, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(c1.grad, c0.grad, atol=3e-3, rtol=3e-4)
    torch.testing.assert_close(w1.grad, w0.grad, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(sw1.grad, sw0.grad, atol=3e-4, rtol=3e-4)


def test_grouped_complex_moe_fused_linear_cuda_fp32_if_available(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("Triton grouped complex MoE CUDA smoke requires CUDA")
    pytest.importorskip("triton")

    from dptb.nn import so2_triton_grouped_linear_ops as ops

    monkeypatch.setenv("DPTB_TRITON_LINEAR_REQUIRE", "1")
    torch.manual_seed(20260423)
    device = torch.device("cuda")
    dtype = torch.float32
    split_sizes = (13, 17, 19, 11)
    n_rows = sum(split_sizes)
    num_experts = 8
    in_features = 37
    out_features = 41

    x0 = torch.randn(n_rows, 2, in_features, device=device, dtype=dtype, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)
    c0 = torch.rand(len(split_sizes), num_experts, device=device, dtype=dtype, requires_grad=True)
    c1 = c0.detach().clone().requires_grad_(True)
    w0 = torch.randn(num_experts, 2 * out_features, in_features, device=device, dtype=dtype, requires_grad=True)
    w1 = w0.detach().clone().requires_grad_(True)

    mixed = torch.einsum("ge,eoi->goi", c0, w0)
    y_ref = _split_loop_complex_linear(x0, mixed, split_sizes)
    y_new = ops.grouped_complex_moe_fused_linear(x1, c1, w1, split_sizes)
    torch.testing.assert_close(y_new, y_ref, atol=2e-4, rtol=2e-4)

    probe = torch.randn_like(y_ref)
    (y_ref * probe).sum().backward()
    (y_new * probe).sum().backward()

    torch.testing.assert_close(x1.grad, x0.grad, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(c1.grad, c0.grad, atol=3e-3, rtol=3e-4)
    torch.testing.assert_close(w1.grad, w0.grad, atol=3e-4, rtol=3e-4)


def test_so2_m_linear_triton_complex_grouped_matches_standard_cpu():
    pytest.importorskip("e3nn")
    from e3nn import o3
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, SO2_m_Linear

    torch.manual_seed(20260423)
    dtype = torch.float64
    split_sizes = (3, 5, 2)
    n_rows = sum(split_sizes)
    num_experts = 5

    coeffs = torch.rand(len(split_sizes), num_experts, dtype=dtype, requires_grad=True)
    coeffs = coeffs / coeffs.sum(dim=-1, keepdim=True)
    coeffs.retain_grad()
    globals_ = MOLEGlobals(coefficients=coeffs, split_sizes=split_sizes)

    irreps_in = o3.Irreps("2x1o + 1x2e + 2x3o")
    irreps_out = o3.Irreps("1x1o + 2x2e + 1x3o")
    standard = SO2_m_Linear(
        1,
        irreps_in,
        irreps_out,
        num_experts=num_experts,
        num_shared_experts=0,
        mole_linear_mode="split_loop",
        so2_m_linear_mode="standard",
    ).to(dtype=dtype)
    triton = SO2_m_Linear(
        1,
        irreps_in,
        irreps_out,
        num_experts=num_experts,
        num_shared_experts=0,
        mole_linear_mode="split_loop",
        so2_m_linear_mode="triton_complex_grouped_linear",
    ).to(dtype=dtype)
    triton.load_state_dict(standard.state_dict(), strict=True)

    x0 = torch.randn(n_rows, 2, standard.num_in_channel, dtype=dtype, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)
    y0 = standard(x0, globals_)
    y1 = triton(x1, globals_)
    torch.testing.assert_close(y1, y0, atol=1e-10, rtol=1e-10)

    probe = torch.randn_like(y0)
    (y0 * probe).sum().backward(retain_graph=True)
    x0_grad = x0.grad.detach().clone()
    coeff_grad = coeffs.grad.detach().clone()
    weight_grads = {
        name: param.grad.detach().clone()
        for name, param in standard.named_parameters()
        if param.grad is not None
    }

    coeffs.grad.zero_()
    (y1 * probe).sum().backward()
    torch.testing.assert_close(x1.grad, x0_grad, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(coeffs.grad, coeff_grad, atol=1e-10, rtol=1e-10)
    for name, param in triton.named_parameters():
        if name in weight_grads:
            torch.testing.assert_close(param.grad, weight_grads[name], atol=1e-10, rtol=1e-10)


def test_so2_m_linear_triton_complex_moe_fused_matches_standard_cpu():
    pytest.importorskip("e3nn")
    from e3nn import o3
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, SO2_m_Linear

    torch.manual_seed(20260423)
    dtype = torch.float64
    split_sizes = (3, 5, 2)
    n_rows = sum(split_sizes)
    num_experts = 5

    coeffs = torch.rand(len(split_sizes), num_experts, dtype=dtype, requires_grad=True)
    coeffs = coeffs / coeffs.sum(dim=-1, keepdim=True)
    coeffs.retain_grad()
    globals_ = MOLEGlobals(coefficients=coeffs, split_sizes=split_sizes)

    irreps_in = o3.Irreps("2x1o + 1x2e + 2x3o")
    irreps_out = o3.Irreps("1x1o + 2x2e + 1x3o")
    standard = SO2_m_Linear(
        1,
        irreps_in,
        irreps_out,
        num_experts=num_experts,
        num_shared_experts=0,
        mole_linear_mode="split_loop",
        so2_m_linear_mode="standard",
    ).to(dtype=dtype)
    fused = SO2_m_Linear(
        1,
        irreps_in,
        irreps_out,
        num_experts=num_experts,
        num_shared_experts=0,
        mole_linear_mode="split_loop",
        so2_m_linear_mode="triton_complex_moe_fused_linear",
    ).to(dtype=dtype)
    fused.load_state_dict(standard.state_dict(), strict=True)

    x0 = torch.randn(n_rows, 2, standard.num_in_channel, dtype=dtype, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)
    y0 = standard(x0, globals_)
    y1 = fused(x1, globals_)
    torch.testing.assert_close(y1, y0, atol=1e-10, rtol=1e-10)

    probe = torch.randn_like(y0)
    (y0 * probe).sum().backward(retain_graph=True)
    x0_grad = x0.grad.detach().clone()
    coeff_grad = coeffs.grad.detach().clone()
    weight_grads = {
        name: param.grad.detach().clone()
        for name, param in standard.named_parameters()
        if param.grad is not None
    }

    coeffs.grad.zero_()
    (y1 * probe).sum().backward()
    torch.testing.assert_close(x1.grad, x0_grad, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(coeffs.grad, coeff_grad, atol=1e-10, rtol=1e-10)
    for name, param in fused.named_parameters():
        if name in weight_grads:
            torch.testing.assert_close(param.grad, weight_grads[name], atol=1e-10, rtol=1e-10)


def test_so2_m_linear_triton_complex_moe_fused_rejects_shared_experts():
    pytest.importorskip("e3nn")
    from e3nn import o3
    from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, SO2_m_Linear

    torch.manual_seed(20260423)
    dtype = torch.float64
    split_sizes = (2, 3)
    num_experts = 4
    coeffs = torch.rand(len(split_sizes), num_experts, dtype=dtype)
    globals_ = MOLEGlobals(coefficients=coeffs, split_sizes=split_sizes)
    layer = SO2_m_Linear(
        1,
        o3.Irreps("1x1o + 1x2e"),
        o3.Irreps("1x1o + 1x2e"),
        num_experts=num_experts,
        num_shared_experts=1,
        mole_linear_mode="split_loop",
        so2_m_linear_mode="triton_complex_moe_fused_linear",
    ).to(dtype=dtype)
    x = torch.randn(sum(split_sizes), 2, layer.num_in_channel, dtype=dtype)

    with pytest.raises(RuntimeError, match="num_shared_experts=0"):
        layer(x, globals_)


def test_mole_linear_env_selects_triton_grouped(monkeypatch):
    from dptb.nn.tensor_product_moe_v3 import MOLELinear

    monkeypatch.setenv("DPTB_MOLE_LINEAR_MODE", "triton_grouped_linear")
    layer = MOLELinear(3, 5, num_experts=2, num_shared_experts=0)

    assert layer.mole_linear_mode == "triton_grouped_linear"


def test_mole_linear_env_selects_triton_fused_expert(monkeypatch):
    from dptb.nn.tensor_product_moe_v3 import MOLELinear

    monkeypatch.setenv("DPTB_MOLE_LINEAR_MODE", "triton_fused_expert_linear")
    layer = MOLELinear(3, 5, num_experts=2, num_shared_experts=0)

    assert layer.mole_linear_mode == "triton_fused_expert_linear"


def test_grouped_linear_require_raises_when_backend_unavailable(monkeypatch):
    from dptb.nn import so2_triton_grouped_linear_ops as ops

    monkeypatch.setenv("DPTB_TRITON_LINEAR_REQUIRE", "1")
    x = torch.randn(4, 3)
    mixed_weights = torch.randn(2, 5, 3)
    split_sizes = (2, 2)

    with pytest.raises(RuntimeError, match="Triton grouped linear backend is unavailable"):
        ops.grouped_linear_apply(x, mixed_weights, None, split_sizes)
