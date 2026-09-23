"""prior_activate (activation-space MoLE) on the SO2CUDA pack/scatter route."""
import pytest
import torch

from dptb.nn import top1_so2_cuda as route
from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, SO2_Linear


def _layer(dtype=torch.float32, device="cpu"):
    torch.manual_seed(20260923)
    return SO2_Linear(
        irreps_in="4x0e + 3x1o + 2x2e",
        irreps_out="4x0e + 3x1o + 2x2e",
        radial_emb=False,
        num_experts=4,
        num_shared_experts=0,
        rotate_in=True,
        rotate_out=True,
        mole_linear_mode="indexed_ref",
        so2_fusion_mode="streamed_m_major_cueq",  # the route that dispatches here
    ).to(device=device, dtype=dtype)


def _inputs(layer, n=37, k=2, device="cpu"):
    g = torch.Generator().manual_seed(7)
    x = torch.randn(n, layer.irreps_in.dim, generator=g).to(device=device, dtype=layer.fc_m0.weight_experts.dtype)
    R = torch.randn(n, 3, generator=g).to(device=device, dtype=x.dtype)
    logits = torch.randn(n, layer.fc_m0.num_experts, generator=g).to(device)
    val, idx = logits.topk(k, dim=-1)
    val = val.softmax(-1).to(x.dtype).detach().requires_grad_(True)
    coeffs = torch.zeros(n, layer.fc_m0.num_experts, device=device, dtype=x.dtype).scatter(1, idx, val)
    globals_ = MOLEGlobals(coefficients=coeffs, sizes=None, topk_indices=idx,
                           topk_values=val, activation_space=True)
    return x.requires_grad_(True), R, globals_


def test_activation_route_skips_cpu_and_can_be_disabled(monkeypatch):
    layer = _layer()
    x, R, g = _inputs(layer)
    assert route.try_activation_forward(layer, x, R, g, route="streamed_m_major_cueq") is None
    monkeypatch.setenv("DPTB_SO2_ACTIVATION_CUDA", "0")
    assert not route.activation_route_enabled()
    monkeypatch.setenv("DPTB_SO2_ACTIVATION_CUDA", "1")
    assert route.activation_route_enabled()


def test_unexpected_cuda_failure_propagates(monkeypatch):
    layer = _layer()
    x, R, g = _inputs(layer)
    monkeypatch.setattr(route, "_ACTIVATION_DISABLED", False)
    monkeypatch.setattr(route, "ACTIVATION_FALLBACKS", 0)

    class _Fake:
        type = "cuda"
    fake_x = type("X", (), {"device": _Fake(), "dtype": torch.float32})()
    monkeypatch.setitem(__import__("sys").modules, "so2_cuda_ops", object())

    def boom(*a, **kw):
        raise RuntimeError("injected CUDA execution failure")
    monkeypatch.setattr(route, "try_forward", boom)
    with pytest.raises(RuntimeError, match="injected CUDA execution failure"):
        route.try_activation_forward(layer, fake_x, R, g, route="streamed_m_major_cueq")
    assert not route._ACTIVATION_DISABLED and route.ACTIVATION_FALLBACKS == 0


def _cuda_route_available():
    if not torch.cuda.is_available():
        return False
    try:
        import so2_cuda_ops  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _cuda_route_available(), reason="needs CUDA and SO2CUDA (so2_cuda_ops)")
def test_cuda_route_matches_streamed_route_forward_and_backward(monkeypatch):
    monkeypatch.setattr(route, "_ACTIVATION_DISABLED", False)
    layer = _layer(device="cuda")
    x, R, g = _inputs(layer, n=211, device="cuda")
    params = [p for p in layer.parameters() if p.requires_grad]

    def run(enabled):
        monkeypatch.setenv("DPTB_SO2_ACTIVATION_CUDA", "1" if enabled else "0")
        calls = route.CALLS
        out, _ = layer(x, R, g)
        grads = torch.autograd.grad(out.square().sum(), [x, g.topk_values, *params])
        return out.detach(), grads, route.CALLS - calls

    ref, ref_grads, ref_calls = run(False)
    got, got_grads, got_calls = run(True)
    assert ref_calls == 0 and got_calls > 0 and not route._ACTIVATION_DISABLED
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)
    for a, b in zip(got_grads, ref_grads):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
