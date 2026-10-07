"""Dense SO(2) API, cache reuse and the rotation backward contract."""
import pytest
import torch
import torch.nn.functional as F

from dptb.nn.tensor_product import SO2_Linear, SO2LinearCached


@pytest.fixture(autouse=True)
def fp64_default():
    previous = torch.get_default_dtype()
    with torch.random.fork_rng(devices=[]):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(421)
        try:
            yield
        finally:
            torch.set_default_dtype(previous)


def test_upstream_tensor_api_matches_explicit_cached_api_and_gradients():
    options = dict(irreps_in="2x0e+2x1o+2x2e", irreps_out="2x0e+2x1o+2x2e",
                   radial_emb=True, latent_dim=4, radial_channels=[5])
    upstream = SO2_Linear(**options)
    cached = SO2LinearCached(**options, so2_m_linear_mode="standard")
    cached.load_state_dict(upstream.state_dict(), strict=True)
    x = torch.randn(6, upstream.irreps_in.dim, requires_grad=True)
    vectors = torch.randn(6, 3, requires_grad=True)
    latents = torch.randn(6, 4, requires_grad=True)
    result = upstream(x, vectors, latents)
    value, rotation = cached(x, vectors, latents)
    assert torch.is_tensor(result)
    assert rotation is not None
    assert torch.equal(result, value)
    gradients = []
    for output, layer in ((result, upstream), (value, cached)):
        gradients.append(torch.autograd.grad(output.square().sum(), (x, vectors, latents, *layer.parameters())))
    for actual, expected in zip(gradients[1], gradients[0]):
        assert torch.isfinite(actual).all()
        assert torch.equal(actual, expected)
    reused, returned_cache = cached(x, vectors, latents, rotation)
    assert returned_cache is rotation
    assert torch.equal(value, reused)


@pytest.mark.parametrize("front", [True, False])
@pytest.mark.parametrize("radial", [True, False])
def test_dense_pair_math_matches_independent_l1_reference(front, radial):
    # One scalar block and one vector block make the SO(2) complex product explicit.
    in_mul, out_mul = (1, 2) if front else (2, 1)
    options = dict(irreps_in=f"2x0e+{in_mul}x1o", irreps_out=f"3x0e+{out_mul}x1o",
                   radial_emb=radial, latent_dim=4, radial_channels=[5],
                   rotate_in=False, rotate_out=False, so2_m_linear_mode="standard")
    layer = SO2LinearCached(**options)
    x = torch.randn(5, layer.irreps_in.dim, requires_grad=True)
    latents = torch.randn(5, 4, requires_grad=True)
    actual, cache = layer(x, None, latents)
    assert cache is None
    scalar, vector = x[:, :2], x[:, 2:].reshape(5, in_mul, 3)
    m0 = torch.cat((scalar, vector[:, :, 1]), dim=-1)
    pair = torch.stack((vector[:, :, 0], vector[:, :, 2]), dim=1)
    weights = layer.radial_emb(latents) if radial else None
    width0 = (2 + in_mul) if front else (3 + out_mul)
    radial0 = None if weights is None else weights[:, :width0]
    radial1 = None if weights is None else weights[:, width0:].unsqueeze(1)
    if radial0 is not None and front:
        m0, pair = m0 * radial0, pair * radial1
    y0 = F.linear(m0, layer.fc_m0.weight, layer.fc_m0.bias)
    raw = F.linear(pair, layer.m_linear[0].fc.weight)
    real, imag = raw[..., :out_mul], raw[..., out_mul:]
    yr, yi = real[:, 0] - imag[:, 1], real[:, 1] + imag[:, 0]
    if radial0 is not None and not front:
        y0 = y0 * radial0
        yr, yi = yr * radial1[:, 0], yi * radial1[:, 0]
    expected = torch.cat((y0[:, :3], torch.stack((yr, y0[:, 3:], yi), dim=-1).flatten(1)), dim=-1)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    inputs = (x, *layer.parameters()) + ((latents,) if radial else ())
    actual_grads = torch.autograd.grad(actual.square().sum(), inputs, retain_graph=True)
    expected_grads = torch.autograd.grad(expected.square().sum(), inputs)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-12, rtol=1e-12)


def test_scalar_only_upstream_layer_has_tensor_output_and_gradients():
    layer = SO2_Linear("2x0e", "3x0e")
    x = torch.randn(4, 2, requires_grad=True)
    result = layer(x, None)
    torch.testing.assert_close(result, F.linear(x, layer.fc_m0.weight, layer.fc_m0.bias), atol=0, rtol=0)
    result.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_cached_interpolation_layer_backpropagates():
    layer = SO2LinearCached("2x0e+2x1o", "2x0e+2x1o", use_interpolation=True)
    x, vectors = torch.randn(4, 8, requires_grad=True), torch.randn(4, 3, requires_grad=True)
    result, rotation = layer(x, vectors)
    assert rotation is not None
    result.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
    assert torch.isfinite(x.grad).all() and torch.isfinite(vectors.grad).all()
