import pytest


def test_e3_element_linear_block_view_matches_indexed_gather_forward_and_grad(monkeypatch):
    torch = pytest.importorskip("torch")
    o3 = pytest.importorskip("e3nn.o3")

    from dptb.nn.rescale import E3ElementLinear

    torch.manual_seed(20260424)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    irreps = o3.Irreps("3x0e + 2x1o + 4x2e + 1x3o")

    monkeypatch.setenv("DPTB_E3_ELEMENT_LINEAR_MODE", "indexed_gather")
    indexed = E3ElementLinear(irreps, device=device, dtype=dtype).to(device=device, dtype=dtype)

    monkeypatch.setenv("DPTB_E3_ELEMENT_LINEAR_MODE", "block_view")
    block = E3ElementLinear(irreps, device=device, dtype=dtype).to(device=device, dtype=dtype)

    n = 11
    x0 = torch.randn(n, irreps.dim, device=device, dtype=dtype, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)
    weights0 = torch.randn(n, indexed.weight_numel, device=device, dtype=dtype, requires_grad=True)
    weights1 = weights0.detach().clone().requires_grad_(True)

    y0 = indexed(x0, weights0)
    y1 = block(x1, weights1)
    torch.testing.assert_close(y1, y0, atol=1e-10, rtol=1e-10)

    probe = torch.randn_like(y0)
    (y0 * probe).sum().backward()
    (y1 * probe).sum().backward()

    torch.testing.assert_close(x1.grad, x0.grad, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(weights1.grad, weights0.grad, atol=1e-10, rtol=1e-10)
