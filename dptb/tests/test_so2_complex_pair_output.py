import pytest

torch = pytest.importorskip("torch")
o3 = pytest.importorskip("e3nn.o3")

from dptb.nn.tensor_product import SO2_m_Linear, complex_pair_output  # noqa: E402


def test_so2_m_linear_is_the_complex_product_of_its_weight_halves():
    torch.manual_seed(20260923)
    # m = 1 keeps the l >= 1 channels: 5 in (2x1o + 3x2e), 3 out (1x1o + 2x3o)
    layer = SO2_m_Linear(1, o3.Irreps("4x0e + 2x1o + 3x2e"), o3.Irreps("2x0e + 1x1o + 2x3o")).double()
    x = torch.randn(7, 2, 5, dtype=torch.float64)

    weight_r, weight_i = layer.fc.weight.detach().split(3, dim=0)
    expected = (weight_r + 1j * weight_i) @ (x[:, 0] + 1j * x[:, 1]).unsqueeze(-1)
    got = layer(x)
    torch.testing.assert_close(got[:, 0], expected.real.squeeze(-1))
    torch.testing.assert_close(got[:, 1], expected.imag.squeeze(-1))


def test_complex_pair_output_gradients():
    x = torch.randn(4, 2, 6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: complex_pair_output(t, 3), (x,))
