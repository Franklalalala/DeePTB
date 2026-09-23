"""prior_activate (activation-space MoLE) on the expert-segmented fused-P0 route."""
import pytest
import torch

from dptb.nn import so2_activation_fused_p0 as fused
from dptb.nn import top1_so2_cuda as pack_scatter
from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, SO2_Linear


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


def _inputs(layer, n=37, k=2, device="cpu", sum_to_one=True):
    g = torch.Generator().manual_seed(7)
    x = torch.randn(n, layer.irreps_in.dim, generator=g).to(device)
    R = torch.randn(n, 3, generator=g).to(device)
    latents = torch.randn(n, 8, generator=g).to(device).requires_grad_(True)
    logits = torch.randn(n, layer.fc_m0.num_experts, generator=g).to(device)
    val, idx = logits.topk(k, dim=-1)
    val = val.softmax(-1) if sum_to_one else val.sigmoid()
    val = val.detach().requires_grad_(True)
    globals_ = _globals(idx, val, layer.fc_m0.num_experts, sum_to_one)
    return x.requires_grad_(True), R, latents, globals_


def _globals(idx, val, num_experts, sum_to_one):
    # as LemMoEV3Edge builds them: without coefficients MOLELinear.forward averages the experts
    coeffs = torch.zeros(idx.shape[0], num_experts, dtype=val.dtype, device=val.device).scatter(1, idx, val)
    return MOLEGlobals(coefficients=coeffs, sizes=None, topk_indices=idx, topk_values=val,
                       activation_space=True, coefficients_sum_to_one=sum_to_one)


def test_route_declines_on_cpu_and_without_routing(monkeypatch):
    layer = _layer("streamed_m_major_fused_p0")
    x, R, latents, g = _inputs(layer)
    assert fused.try_forward(layer, x, R, g) is None
    assert fused._routed(g, x)
    assert not fused._routed(MOLEGlobals(sizes=None, topk_indices=g.topk_indices, topk_values=g.topk_values,
                                         activation_space=True), x)
    monkeypatch.setenv("DPTB_SO2_ACTIVATION_FUSED_P0", "0")
    assert not fused.enabled()
    monkeypatch.setenv("DPTB_SO2_ACTIVATION_FUSED_P0", "1")
    assert fused.enabled()


def test_cpu_fused_p0_config_runs_the_streamed_route():
    ref = _layer("streamed_m_major_cueq")
    got = _layer("streamed_m_major_fused_p0")
    got.load_state_dict(ref.state_dict())
    x, R, latents, g = _inputs(ref)
    torch.testing.assert_close(got(x, R, g)[0], ref(x, R, g)[0], rtol=0, atol=0)


def test_prior_activate_accepts_fused_p0():
    from dptb.tests.test_lem_moe_v3_prior_2b import _build

    m = _build(False, edge_router_prior_activate=True, num_experts=4, num_shared_experts=1, top_k=2,
               so2_fusion_mode="streamed_m_major_fused_p0")
    routes = {getattr(mod, "so2_fusion_mode") for mod in m.embedding.modules() if hasattr(mod, "so2_fusion_mode")}
    assert routes == {"streamed_m_major_fused_p0"}


def _cuda_route_available():
    if not torch.cuda.is_available():
        return False
    try:
        import so2_cuda_ops  # noqa: F401
    except ImportError:
        return False
    return True


CASES = [
    dict(),                                   # front, shared expert folded
    dict(irreps_out="3x0e + 2x1o + 1x2e"),    # radial after the linear (front False)
    dict(radial=True),
    dict(radial=True, irreps_out="3x0e + 2x1o + 1x2e"),
    dict(interpolation=True),                 # m>0 blocks are not MoLE linears
    dict(shared=0),
    dict(shared=2),
    dict(radial=True, interpolation=True),
    dict(radial=True, interpolation=True, irreps_out="3x0e + 2x1o + 1x2e"),
    dict(rotate_in=False, rotate_out=False),
    dict(rotate_in=False),
    dict(rotate_out=False),
]


@pytest.mark.skipif(not _cuda_route_available(), reason="needs CUDA and SO2CUDA (so2_cuda_ops)")
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("schedule", ["expanded", "per_slot"])
@pytest.mark.parametrize("sum_to_one", [True, False])
def test_fused_route_matches_streamed_route(monkeypatch, case, schedule, sum_to_one):
    monkeypatch.setattr(fused, "_DISABLED", False)
    monkeypatch.setattr(pack_scatter, "_ACTIVATION_DISABLED", False)
    monkeypatch.setenv("DPTB_SO2_ACTIVATION_FUSED_P0_GEMM", schedule)
    ref_layer = _layer("streamed_m_major_cueq", device="cuda", **case)
    layer = _layer("streamed_m_major_fused_p0", device="cuda", **case)
    layer.load_state_dict(ref_layer.state_dict())
    x, R, latents, g = _inputs(layer, n=211, device="cuda", sum_to_one=sum_to_one)
    lat = latents if case.get("radial") else None

    def run(mod, activation_cuda):
        monkeypatch.setenv("DPTB_SO2_ACTIVATION_CUDA", "1" if activation_cuda else "0")
        g_run = _globals(g.topk_indices, g.topk_values, layer.fc_m0.num_experts, sum_to_one)
        calls = (fused.CALLS, pack_scatter.CALLS)
        out, _ = mod(x, R, g_run, latents=lat)
        params = [p for p in mod.parameters() if p.requires_grad]
        grads = torch.autograd.grad(out.square().sum(), [x, g.topk_values, *([lat] if lat is not None else []), *params])
        return out.detach(), grads, (fused.CALLS - calls[0], pack_scatter.CALLS - calls[1])

    ref, ref_grads, ref_calls = run(ref_layer, False)
    got, got_grads, got_calls = run(layer, False)
    assert ref_calls == (0, 0) and got_calls == (1, 0) and not fused._DISABLED
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)
    for a, b in zip(got_grads, ref_grads):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
    # the pack/scatter route (fused route off) agrees as well
    monkeypatch.setenv("DPTB_SO2_ACTIVATION_FUSED_P0", "0")
    ps, ps_grads, ps_calls = run(layer, True)
    assert ps_calls == (0, 1)
    torch.testing.assert_close(ps, got, rtol=1e-5, atol=1e-5)
    for a, b in zip(ps_grads, got_grads):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)


SWITCH_CASES = [dict(), dict(irreps_out="3x0e + 2x1o + 1x2e"), dict(radial=True), dict(interpolation=True)]


@pytest.mark.skipif(not _cuda_route_available(), reason="needs CUDA and SO2CUDA (so2_cuda_ops)")
@pytest.mark.parametrize("case", SWITCH_CASES)
@pytest.mark.parametrize("schedule", ["per_slot", "expanded"])
def test_fused_route_matches_switch_top1(monkeypatch, case, schedule):
    from dptb.nn.top1_prior import Top1Route

    monkeypatch.setenv("DPTB_SO2_ACTIVATION_FUSED_P0_GEMM", schedule)
    monkeypatch.setattr(fused, "_DISABLED", False)
    monkeypatch.setattr(pack_scatter, "_ACTIVATION_DISABLED", False)
    ref_layer = _layer("streamed_m_major_cueq", device="cuda", shared=0, **case)
    layer = _layer("streamed_m_major_fused_p0", device="cuda", shared=0, **case)
    layer.load_state_dict(ref_layer.state_dict())
    x, R, latents, g = _inputs(layer, n=211, k=1, device="cuda")
    lat = latents if case.get("radial") else None
    ids = g.topk_indices
    gates = torch.rand(ids.shape, generator=torch.Generator().manual_seed(3)).to("cuda").requires_grad_(True)

    def run(mod, reference):
        route = Top1Route(ids, gates)
        route.top1_reference_so2 = reference
        calls = (fused.CALLS, fused.TOP1_CALLS, pack_scatter.CALLS)
        out, _ = mod(x, R, route, latents=lat)
        params = [p for p in mod.parameters() if p.requires_grad]
        grads = torch.autograd.grad(out.square().sum(), [x, gates, *([lat] if lat is not None else []), *params])
        return out.detach(), grads, (fused.CALLS - calls[0], fused.TOP1_CALLS - calls[1], pack_scatter.CALLS - calls[2])

    ref, ref_grads, ref_calls = run(ref_layer, True)  # streamed route, no SO2CUDA
    got, got_grads, got_calls = run(layer, False)
    assert ref_calls == (0, 0, 0) and got_calls == (1, 1, 0) and not fused._DISABLED
    torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)
    for a, b in zip(got_grads, ref_grads):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
    # the top-1 pack/scatter branch of the grouped streaming route agrees as well
    ps, ps_grads, ps_calls = run(ref_layer, False)
    assert ps_calls == (0, 0, 1)
    torch.testing.assert_close(ps, got, rtol=1e-5, atol=1e-5)
    for a, b in zip(ps_grads, got_grads):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
