"""Activation-space SO2 layers (prior_activate, Switch top-1) on the SO2CUDA routes."""
import pytest
import torch

from dptb.nn import so2_activation_routes as routes
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


def _calls():
    return (routes.STATS.calls[routes.FUSED_P0], routes.STATS.calls[routes.FUSED_P0 + "_top1"],
            routes.STATS.calls[routes.PACK_SCATTER])


def test_cpu_calls_decline_and_run_the_streamed_route():
    ref = _layer("streamed_m_major_cueq")
    got = _layer("streamed_m_major_fused_p0")
    got.load_state_dict(ref.state_dict())
    x, R, _, g = _inputs(ref)
    before = routes.STATS.declines[("all", "not CUDA float32")]
    torch.testing.assert_close(got(x, R, g)[0], ref(x, R, g)[0], rtol=0, atol=0)
    assert routes.STATS.declines[("all", "not CUDA float32")] == before + 2
    assert routes.forward(ref, x, R, MOLEGlobals(sizes=None), fused=True) is None
    assert routes.STATS.declines[("all", "weight-space routing")] > 0


def test_fused_route_needs_per_row_coefficients():
    layer = _layer("streamed_m_major_fused_p0")
    x, _, _, g = _inputs(layer)
    assert routes._per_row_routing(g, x)
    no_coefficients = MOLEGlobals(sizes=None, topk_indices=g.topk_indices, topk_values=g.topk_values,
                                  activation_space=True)
    assert not routes._per_row_routing(no_coefficients, x)
    assert not routes._per_row_routing(g, x[:-1])


def test_route_switches(monkeypatch):
    monkeypatch.setenv("DPTB_SO2_ACTIVATION_FUSED_P0", "0")
    assert not routes._switch_on("DPTB_SO2_ACTIVATION_FUSED_P0")
    monkeypatch.setenv("DPTB_SO2_ACTIVATION_FUSED_P0_GEMM", "expanded")
    assert routes._gemm_schedule() == "expanded"
    monkeypatch.setenv("DPTB_SO2_ACTIVATION_FUSED_P0_GEMM", "all")
    with pytest.raises(RuntimeError, match="DPTB_SO2_ACTIVATION_FUSED_P0_GEMM"):
        routes._gemm_schedule()


def test_error_inside_a_route_propagates(monkeypatch):
    layer = _layer("streamed_m_major_fused_p0")
    x, R, _, g = _inputs(layer)
    monkeypatch.setattr(routes, "_preflight", lambda *a: None)
    monkeypatch.setattr(routes, "_load_ops", lambda: (object(), object()))
    monkeypatch.setattr(routes, "_wigner_layout", lambda ops, module, x, R, w: (w, (None, None, 0, 0)))

    def boom(*a, **kw):
        raise RuntimeError("injected CUDA execution failure")

    monkeypatch.setattr(routes, "_Packed", boom)
    calls = _calls()
    with pytest.raises(RuntimeError, match="injected CUDA execution failure"):
        layer(x, R, g)
    assert _calls() == calls


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
@pytest.mark.parametrize("schedule", ["expanded", "per_slot"])
@pytest.mark.parametrize("sum_to_one", [True, False])
def test_prior_activate_routes_match_streamed_route(monkeypatch, case, schedule, sum_to_one):
    monkeypatch.setenv("DPTB_SO2_ACTIVATION_FUSED_P0_GEMM", schedule)
    ref_layer = _layer("streamed_m_major_cueq", device="cuda", **case)
    layer = _layer("streamed_m_major_fused_p0", device="cuda", **case)
    layer.load_state_dict(ref_layer.state_dict())
    x, R, latents, g = _inputs(layer, n=211, device="cuda", sum_to_one=sum_to_one)
    lat = latents if case.get("radial") else None

    def run(mod, activation_cuda):
        monkeypatch.setenv("DPTB_SO2_ACTIVATION_CUDA", "1" if activation_cuda else "0")
        route = _globals(g.topk_indices, g.topk_values, layer.fc_m0.num_experts, sum_to_one)
        calls = _calls()
        result = _run(mod, x, R, route, lat, g.topk_values)
        return result, tuple(b - a for a, b in zip(calls, _calls()))

    ref, ref_calls = run(ref_layer, False)          # grouped streaming route
    got, got_calls = run(layer, False)              # fused-P0
    assert ref_calls == (0, 0, 0) and got_calls == (1, 0, 0)
    _assert_same(got, ref)
    monkeypatch.setenv("DPTB_SO2_ACTIVATION_FUSED_P0", "0")
    ps, ps_calls = run(layer, True)                 # pack/scatter
    assert ps_calls == (0, 0, 1)
    _assert_same(ps, got)


@requires_so2_cuda
def test_inference_warmup_then_training():
    """Layouts cached under torch.inference_mode must still be usable by a training step."""
    layer = _layer("streamed_m_major_fused_p0", device="cuda", radial=True)
    x, R, latents, g = _inputs(layer, n=64, device="cuda")
    with torch.inference_mode():
        layer(x.detach(), R, _globals(g.topk_indices, g.topk_values.detach(), layer.fc_m0.num_experts, True),
              latents=latents.detach())
    calls = _calls()
    _, grads = _run(layer, x, R, _globals(g.topk_indices, g.topk_values, layer.fc_m0.num_experts, True),
                    latents, g.topk_values)
    assert _calls()[0] == calls[0] + 1
    assert all(torch.isfinite(grad).all() for grad in grads)


SWITCH_CASES = {"front": dict(), "back": dict(irreps_out="3x0e + 2x1o + 1x2e"), "radial": dict(radial=True),
                "interp": dict(interpolation=True)}


@requires_so2_cuda
@pytest.mark.parametrize("case", SWITCH_CASES.values(), ids=SWITCH_CASES.keys())
@pytest.mark.parametrize("schedule", ["per_slot", "expanded"])
def test_switch_routes_match_streamed_route(monkeypatch, case, schedule):
    from dptb.nn.top1_prior import Top1Route

    monkeypatch.setenv("DPTB_SO2_ACTIVATION_FUSED_P0_GEMM", schedule)
    ref_layer = _layer("streamed_m_major_cueq", device="cuda", shared=0, **case)
    layer = _layer("streamed_m_major_fused_p0", device="cuda", shared=0, **case)
    layer.load_state_dict(ref_layer.state_dict())
    x, R, latents, g = _inputs(layer, n=211, k=1, device="cuda")
    lat = latents if case.get("radial") else None
    gates = torch.rand(g.topk_indices.shape, generator=torch.Generator().manual_seed(3)).to("cuda").requires_grad_(True)

    def run(mod, reference):
        route = Top1Route(g.topk_indices, gates)
        route.top1_reference_so2 = reference
        calls = _calls()
        result = _run(mod, x, R, route, lat, gates)
        return result, tuple(b - a for a, b in zip(calls, _calls()))

    ref, ref_calls = run(ref_layer, True)           # streamed route, no SO2CUDA
    got, got_calls = run(layer, False)              # fused-P0
    assert ref_calls == (0, 0, 0) and got_calls == (1, 1, 0)
    _assert_same(got, ref)
    ps, ps_calls = run(ref_layer, False)            # pack/scatter
    assert ps_calls == (0, 0, 1)
    _assert_same(ps, got)
