"""Activation-space SO2 layers (prior_activate, Switch top-1) on the SO2CUDA routes."""
import pytest
import torch

from dptb.nn import so2_backend as routes
from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, SO2_Linear
from dptb.tests._requires import requires_so2_cuda


def _layer(route, *, irreps_out="4x0e + 3x1o + 2x2e", radial=False, interpolation=False, shared=1,
           rotate_in=True, rotate_out=True, device="cpu"):
    torch.manual_seed(20260923)
    layer = SO2_Linear(
        irreps_in="4x0e + 3x1o + 2x2e",
        irreps_out=irreps_out,
        radial_emb=radial,
        latent_dim=8 if radial else None,
        radial_channels=[16] if radial else None,
        use_interpolation=interpolation,
        num_experts=4,
        num_shared_experts=shared,
        rotate_in=rotate_in,
        rotate_out=rotate_out,
        mole_linear_mode="indexed_ref",
        so2_fusion_mode=route,
    )
    return layer.to(device=device, dtype=torch.float32)


def _globals(idx, val, num_experts, sum_to_one):
    # as LemMoEV3Edge builds them: without coefficients MOLELinear.forward averages the experts
    coeffs = torch.zeros(idx.shape[0], num_experts, dtype=val.dtype, device=val.device).scatter(1, idx, val)
    return MOLEGlobals(coefficients=coeffs, sizes=None, topk_indices=idx, topk_values=val,
                       activation_space=True, coefficients_sum_to_one=sum_to_one)


def _inputs(layer, n=37, k=2, device="cpu", sum_to_one=True):
    g = torch.Generator().manual_seed(7)
    x = torch.randn(n, layer.irreps_in.dim, generator=g).to(device)
    R = torch.randn(n, 3, generator=g).to(device)
    latents = torch.randn(n, 8, generator=g).to(device).requires_grad_(True)
    logits = torch.randn(n, layer.fc_m0.num_experts, generator=g).to(device)
    val, idx = logits.topk(k, dim=-1)
    val = (val.softmax(-1) if sum_to_one else val.sigmoid()).detach().requires_grad_(True)
    return x.requires_grad_(True), R, latents, _globals(idx, val, layer.fc_m0.num_experts, sum_to_one)


def test_cpu_fused_configuration_uses_reference():
    ref = _layer("streamed_m_major_ref")
    got = _layer("streamed_m_major_fused_p0")
    got.load_state_dict(ref.state_dict())
    x, R, _, g = _inputs(ref)
    torch.testing.assert_close(got(x, R, g)[0], ref(x, R, g)[0], rtol=0, atol=0)


def test_disabled_backend_grouped_linear_and_gradients(monkeypatch, caplog):
    monkeypatch.setenv("SO2_CUDA_BACKEND", "off")
    x = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    w = torch.randn(3, 4, 3, dtype=torch.float64, requires_grad=True)
    ptr = torch.tensor([0, 2, 2, 5])
    got = routes.grouped_gemm(x, ptr, w)
    ref = torch.cat([torch.nn.functional.linear(x[:2], w[0]),
                     torch.nn.functional.linear(x[2:2], w[1]),
                     torch.nn.functional.linear(x[2:], w[2])])
    torch.testing.assert_close(got, ref, rtol=0, atol=0)
    probe = torch.randn_like(got)
    actual = torch.autograd.grad((got * probe).sum(), (x, w))
    expected = torch.autograd.grad((ref * probe).sum(), (x, w))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("ptr", [[1, 2, 5], [0, 6, 5], [0, 2, 4]])
def test_grouped_reference_rejects_invalid_segments(monkeypatch, ptr):
    monkeypatch.setenv("SO2_CUDA_BACKEND", "off")
    with pytest.raises(ValueError):
        routes.grouped_gemm(torch.randn(5, 3), torch.tensor(ptr), torch.randn(2, 4, 3))


def test_backend_execution_errors_propagate(monkeypatch):
    from types import SimpleNamespace

    def fail(*args, **kwargs):
        raise RuntimeError("injected execution failure")

    monkeypatch.setattr(routes, "backend", lambda: SimpleNamespace(grouped_gemm=fail))
    monkeypatch.setattr(routes, "_cuda_fp32", lambda *args: True)
    with pytest.raises(RuntimeError, match="injected execution failure"):
        routes.grouped_gemm(torch.randn(5, 3), torch.tensor([0, 5]), torch.randn(1, 4, 3))


CASES = {
    "front": dict(),                                  # radial before the linear, shared expert folded
    "back": dict(irreps_out="3x0e + 2x1o + 1x2e"),    # radial after the linear
    "radial": dict(radial=True),
    "radial-back": dict(radial=True, irreps_out="3x0e + 2x1o + 1x2e"),
    "interp": dict(interpolation=True),               # m>0 blocks are not MoLE linears
    "shared0": dict(shared=0),
    "shared2": dict(shared=2),
    "radial-interp": dict(radial=True, interpolation=True),
    "radial-interp-back": dict(radial=True, interpolation=True, irreps_out="3x0e + 2x1o + 1x2e"),
    "no-rotation": dict(rotate_in=False, rotate_out=False),
    "no-rotate-in": dict(rotate_in=False),
    "no-rotate-out": dict(rotate_out=False),
}


def _run(layer, x, R, route, lat, gate):
    out, _ = layer(x, R, route, latents=lat)
    params = [p for p in layer.parameters() if p.requires_grad]
    grads = torch.autograd.grad(out.square().sum(), [x, gate, *([lat] if lat is not None else []), *params])
    return out.detach(), grads


def _assert_same(got, ref):
    torch.testing.assert_close(got[0], ref[0], rtol=1e-5, atol=1e-5)
    for a, b in zip(got[1], ref[1]):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)


@requires_so2_cuda
@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
@pytest.mark.parametrize("sum_to_one", [True, False])
def test_activation_cuda_matches_reference(monkeypatch, case, sum_to_one):
    ref_layer = _layer("streamed_m_major_ref", device="cuda", **case)
    layer = _layer("streamed_m_major_fused_p0", device="cuda", **case)
    layer.load_state_dict(ref_layer.state_dict())
    x, R, latents, g = _inputs(layer, n=211, device="cuda", sum_to_one=sum_to_one)
    lat = latents if case.get("radial") else None
    ref = _run(ref_layer, x, R, g, lat, g.topk_values)
    calls = routes.STATS.calls["fused_p0"]
    got = _run(layer, x, R, g, lat, g.topk_values)
    assert routes.STATS.calls["fused_p0"] == calls + (0 if case.get("interpolation") else 1)
    _assert_same(got, ref)


@requires_so2_cuda
def test_inference_warmup_then_training():
    """Integer layouts cached during inference remain usable by autograd."""
    layer = _layer("streamed_m_major_fused_p0", device="cuda", radial=True)
    x, R, latents, g = _inputs(layer, n=64, device="cuda")
    with torch.inference_mode():
        layer(x.detach(), R, _globals(g.topk_indices, g.topk_values.detach(), layer.fc_m0.num_experts, True),
              latents=latents.detach())
    _, grads = _run(layer, x, R, _globals(g.topk_indices, g.topk_values, layer.fc_m0.num_experts, True),
                    latents, g.topk_values)
    assert all(torch.isfinite(grad).all() for grad in grads)
