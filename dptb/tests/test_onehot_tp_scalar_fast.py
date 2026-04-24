import pytest


def _assert_forward_and_grad_close(torch, tp_ref, tp_fast, x, y):
    x_ref = x.detach().clone().requires_grad_(True)
    y_ref = y.detach().clone().requires_grad_(True)
    x_fast = x.detach().clone().requires_grad_(True)
    y_fast = y.detach().clone().requires_grad_(True)

    out_ref = tp_ref(x_ref, y_ref)
    out_fast = tp_fast(x_fast, y_fast)

    torch.testing.assert_close(out_fast, out_ref, atol=1e-10, rtol=1e-10)

    loss_ref = out_ref.square().sum()
    loss_fast = out_fast.square().sum()
    loss_ref.backward()
    loss_fast.backward()

    torch.testing.assert_close(x_fast.grad, x_ref.grad, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(y_fast.grad, y_ref.grad, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(tp_fast.weight.grad, tp_ref.weight.grad, atol=1e-10, rtol=1e-10)


def test_scalar_fast_matches_uvu_tensor_product_forward_and_grad():
    torch = pytest.importorskip("torch")
    o3 = pytest.importorskip("e3nn.o3")
    from dptb.nn.embedding.lem_moe_v3 import ScalarOnehotTP

    torch.manual_seed(20260424)
    dtype = torch.float64
    irreps = o3.Irreps("3x0e + 2x1o + 1x2e")
    onehot_irreps = o3.Irreps("7x0e")
    instructions = [(i, 0, i, "uvu", True) for i, _ in enumerate(irreps)]

    tp_ref = o3.TensorProduct(irreps, onehot_irreps, irreps, instructions).to(dtype=dtype)
    tp_fast = ScalarOnehotTP.from_e3nn(tp_ref).to(dtype=dtype)

    x = torch.randn(11, irreps.dim, dtype=dtype)
    y = torch.randn(11, onehot_irreps.dim, dtype=dtype)

    _assert_forward_and_grad_close(torch, tp_ref, tp_fast, x, y)


def test_scalar_fast_packs_diagonal_uvu_forward_without_einsum_or_cat(monkeypatch):
    torch = pytest.importorskip("torch")
    o3 = pytest.importorskip("e3nn.o3")
    from dptb.nn.embedding.lem_moe_v3 import ScalarOnehotTP

    torch.manual_seed(20260424)
    dtype = torch.float64
    irreps = o3.Irreps("2x0e + 1x1o + 3x2e + 4x3o")
    onehot_irreps = o3.Irreps("5x0e")
    instructions = [(i, 0, i, "uvu", True) for i, _ in enumerate(irreps)]

    tp_ref = o3.TensorProduct(irreps, onehot_irreps, irreps, instructions).to(dtype=dtype)
    tp_fast = ScalarOnehotTP.from_e3nn(tp_ref).to(dtype=dtype)

    x = torch.randn(7, irreps.dim, dtype=dtype)
    y = torch.randn(7, onehot_irreps.dim, dtype=dtype)
    out_ref = tp_ref(x, y)

    def _fail_einsum(*args, **kwargs):
        raise AssertionError("packed diagonal uvu path should not call torch.einsum")

    def _fail_cat(*args, **kwargs):
        raise AssertionError("packed diagonal uvu path should not cat per-path weights")

    monkeypatch.setattr(torch, "einsum", _fail_einsum)
    monkeypatch.setattr(torch, "cat", _fail_cat)

    out_fast = tp_fast(x, y)
    torch.testing.assert_close(out_fast, out_ref, atol=1e-10, rtol=1e-10)


def test_scalar_fast_matches_fully_connected_scalar_tp_forward_and_grad():
    torch = pytest.importorskip("torch")
    o3 = pytest.importorskip("e3nn.o3")
    from dptb.nn.embedding.lem_moe_v3 import ScalarOnehotTP

    torch.manual_seed(20260424)
    dtype = torch.float64
    irreps_in = o3.Irreps("3x0e + 2x1o + 2x2e")
    irreps_out = o3.Irreps("2x0e + 3x1o + 1x2e")
    onehot_irreps = o3.Irreps("5x0e")

    tp_ref = o3.FullyConnectedTensorProduct(irreps_in, onehot_irreps, irreps_out).to(dtype=dtype)
    tp_fast = ScalarOnehotTP.from_e3nn(tp_ref).to(dtype=dtype)

    x = torch.randn(13, irreps_in.dim, dtype=dtype)
    y = torch.randn(13, onehot_irreps.dim, dtype=dtype)

    _assert_forward_and_grad_close(torch, tp_ref, tp_fast, x, y)
