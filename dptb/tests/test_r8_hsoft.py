"""Full-soft dispatch: independent expert sum, gradients and restart contracts."""
import copy

import pytest
import torch
from torch.nn import functional as F

from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLELinear, MOLERouterV3, SO2_Linear
from dptb.tests._requires import requires_so2_cuda
from dptb.tests.model_helpers import _build, _data


def _route(router, features):
    coeff, _, _ = router(features)
    idx, val = router.last_topk()
    return MOLEGlobals(coefficients=coeff, topk_indices=idx, topk_values=val,
                       activation_space=True, coefficients_sum_to_one=router.coefficients_sum_to_one)


def _check(a, b, record_property, label, atol=1e-10, rtol=0):
    error = float((a.detach().double() - b.detach().double()).abs().max()) if a.numel() else 0.0
    record_property(label, error)
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


@pytest.mark.parametrize("pair_rows", [False, True], ids=["m0_bias", "m_positive"])
def test_full_soft_explicit_sum_and_all_gradients(pair_rows, record_property):
    torch.manual_seed(812)
    router = MOLERouterV3(9, num_experts=4, top_k=4).double()
    lin = MOLELinear(7, 5, num_experts=4, num_shared_experts=1, bias=not pair_rows,
                     mole_expert_parameterization="shared_core", mole_expert_rank=3,
                     mole_linear_mode="split_loop").double()
    x = torch.randn((13, 2, 7) if pair_rows else (13, 7), dtype=torch.float64, requires_grad=True)
    features = torch.randn(13, 9, dtype=torch.float64, requires_grad=True)
    got = lin(x, _route(router, features))
    probs = router.net(features).softmax(-1)
    ref = F.linear(x, lin.weight_shared[0], None if pair_rows else lin.bias_shared[0])
    for e in range(4):
        weight = lin.basis_left @ lin.core_experts[e] @ lin.basis_right.t()
        out = F.linear(x, weight, None if pair_rows else lin.bias_experts[e])
        ref = ref + out * probs[:, e].reshape(13, *([1] * (x.dim() - 1)))
    _check(got, ref, record_property, "fp64_forward")
    named = [("input", x), ("router_input", features)]
    named += [("router." + k, v) for k, v in router.named_parameters()]
    named += list(lin.named_parameters())
    cotangent = torch.randn_like(got)
    ga = torch.autograd.grad((got * cotangent).sum(), [v for _, v in named])
    gb = torch.autograd.grad((ref * cotangent).sum(), [v for _, v in named])
    for (name, _), a, b in zip(named, ga, gb):
        _check(a, b, record_property, "fp64_grad_" + name)
        assert torch.isfinite(a).all()
    assert all(g.abs().sum() > 0 for g in ga)


@pytest.mark.parametrize("top_k", [4, None])
@pytest.mark.parametrize("fast", [True, False])
@pytest.mark.parametrize("gate", ["renorm", "full_softmax"])
@pytest.mark.parametrize("n", [0, 11])
def test_full_soft_metadata_stats_and_frozen_bias(top_k, fast, gate, n):
    torch.manual_seed(72)
    router = MOLERouterV3(7, num_experts=4, top_k=top_k, full_expert_fast_path=fast,
                          gate=gate, select_noise=1.0, bias_update_speed=0.4).double()
    router.expert_bias.copy_(torch.tensor([2., 1., -1., 0.]))
    router.record_train_stats = True
    before = {k: v.clone() for k, v in router.state_dict().items()}
    x = torch.randn(n, 7, dtype=torch.float64, requires_grad=True)
    coeff, monitor, cv = router(x)
    idx, val = router.last_topk()
    assert torch.equal(idx, torch.arange(4).expand(n, -1))
    assert val is coeff and val.requires_grad and val.shape == (n, 4)
    assert router.coefficients_sum_to_one
    assert torch.equal(coeff, router.net(x).softmax(-1))
    assert torch.isfinite(monitor) and cv == 0 and torch.isfinite(router.last_router_z_loss)
    assert torch.equal(before["expert_bias"], router.expert_bias)
    if fast or top_k is None:
        assert torch.equal(before["ema_load"], router.ema_load)
    else:
        torch.testing.assert_close(router.ema_load, before["ema_load"] * 0.9 + n * 0.1)
    stats = router.last_train_stats
    torch.testing.assert_close(stats["soft_load"], coeff.detach().float().sum(0))
    torch.testing.assert_close(stats["soft_load_sq"], coeff.detach().float().square().sum(0))
    assert torch.equal(stats["hard_load"], torch.full((4,), float(n)))
    assert torch.isfinite(stats["mmp"])
    router.eval()(x)
    assert router.last_train_stats is stats


def test_empty_edge_dispatch_clears_previous_metadata():
    model = _build(False, num_experts=4, top_k=4, edge_router_prior_activate=True,
                   mole_expert_parameterization="shared_core", mole_expert_rank=3,
                   so2_fusion_mode="staged")
    emb = model.embedding
    emb.router.record_train_stats = True
    emb._make_edge_moe_globals(torch.randn(3, emb.edge_router_in_features), torch.zeros(3, dtype=torch.long))
    route, monitor, cv, count = emb._make_edge_moe_globals(
        torch.empty(0, emb.edge_router_in_features), torch.empty(0, dtype=torch.long))
    assert route.activation_space and route.coefficients_sum_to_one
    assert route.topk_indices.shape == route.topk_values.shape == route.coefficients.shape == (0, 4)
    assert route.topk_values is emb.router.last_topk()[1]
    assert all(value == 0 for value in (monitor, cv, count))
    assert emb.router.last_train_stats["n_rows"] == 0
    for pair in (False, True):
        lin = MOLELinear(7, 5, num_experts=4, mole_expert_parameterization="shared_core", mole_expert_rank=3)
        shape = (0, 2, 7) if pair else (0, 7)
        out = lin(torch.empty(shape), route)
        assert out.shape == (*shape[:-1], 5) and torch.isfinite(out).all()


@pytest.mark.parametrize("gate", ["renorm", "full_softmax"])
def test_k2_coefficients_gradients_and_bias_follow_legacy_equations(gate):
    torch.manual_seed(58)
    router = MOLERouterV3(7, num_experts=4, top_k=2, gate=gate).double()
    x = torch.randn(11, 7, dtype=torch.float64, requires_grad=True)
    bias, ema = router.expert_bias.clone(), router.ema_load.clone()
    logits = router.net(x)
    idx = (logits.sigmoid() + bias).topk(2, -1).indices
    val = logits.gather(1, idx).softmax(-1) if gate == "renorm" else logits.softmax(-1).gather(1, idx)
    expected = torch.zeros_like(logits).scatter(1, idx, val)
    coeff, _, _ = router(x)
    assert torch.equal(router.last_topk()[0], idx) and torch.equal(coeff, expected)
    load = F.one_hot(idx, num_classes=4).float().sum((0, 1))
    new_bias = bias - (load - 11 * 2 / 4).sign() * router.bias_update_speed
    new_bias -= new_bias.mean()
    assert torch.equal(router.expert_bias, new_bias)
    assert torch.equal(router.ema_load, ema.mul_(0.9).add_(load, alpha=0.1))
    args = [x, *router.parameters()]
    a = torch.autograd.grad(coeff.square().sum(), args)
    b = torch.autograd.grad(expected.square().sum(), args)
    assert all(torch.equal(aa, bb) for aa, bb in zip(a, b))


def test_model_strict_load_k2_state_into_hsoft_and_resume_one_step(tmp_path, record_property):
    opts = dict(num_experts=4, num_shared_experts=1, edge_router_prior_activate=True,
                mole_expert_parameterization="shared_core", mole_expert_rank=3, so2_fusion_mode="staged")
    legacy = _build(False, top_k=2, **opts)
    model = _build(False, top_k=4, **opts)
    model.load_state_dict(legacy.state_dict(), strict=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    data = _data(model)

    def step(m, o):
        o.zero_grad(set_to_none=True)
        out = m(copy.deepcopy(data))
        loss = out["node_features"].square().mean() + out["edge_features"].square().mean()
        loss.backward()
        o.step()
        assert torch.isfinite(loss)
        return out

    step(model, opt)
    path = tmp_path / "hsoft.pth"
    torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict()), path)
    other = _build(False, top_k=4, **opts)
    other_opt = torch.optim.Adam(other.parameters(), lr=1e-3)
    checkpoint = torch.load(path, weights_only=True)
    other.load_state_dict(checkpoint["model"], strict=True)
    other_opt.load_state_dict(checkpoint["optimizer"])
    a, b = step(model, opt), step(other, other_opt)
    for key in ("node_features", "edge_features"):
        assert torch.equal(a[key], b[key])
    for k, v in model.state_dict().items():
        _check(v, other.state_dict()[k], record_property, "fp32_resume_" + k, 1e-6, 1e-5)


def test_so2_m_positive_four_expert_sum_fp64(record_property):
    torch.manual_seed(907)
    lin = SO2_Linear(irreps_in="4x0e+3x1o+2x2e", irreps_out="3x0e+2x1o+2x2e",
                     num_experts=4, num_shared_experts=1, mole_expert_parameterization="shared_core",
                     mole_expert_rank=3, mole_linear_mode="split_loop", so2_fusion_mode="staged").double()
    router = MOLERouterV3(9, num_experts=4, top_k=4).double()
    x = torch.randn(11, lin.irreps_in.dim, dtype=torch.float64, requires_grad=True)
    h = torch.randn(11, 9, dtype=torch.float64, requires_grad=True)
    r = torch.randn(11, 3, dtype=torch.float64)
    got = lin(x, r, _route(router, h))[0]
    probs = router.net(h).softmax(-1)
    ref = torch.zeros_like(got)
    for e in range(4):
        idx = torch.full((11, 1), e, dtype=torch.long)
        val = torch.ones(11, 1, dtype=torch.float64)
        route = MOLEGlobals(coefficients=F.one_hot(idx[:, 0], 4).double(), topk_indices=idx,
                            topk_values=val, activation_space=True, coefficients_sum_to_one=True)
        ref = ref + probs[:, e:e+1] * lin(x, r, route)[0]
    _check(got, ref, record_property, "fp64_so2_forward")
    args = [x, h, *router.parameters(), *lin.parameters()]
    ga = torch.autograd.grad(got.square().sum(), args)
    gb = torch.autograd.grad(ref.square().sum(), args)
    for i, (a, b) in enumerate(zip(ga, gb)):
        _check(a, b, record_property, "fp64_so2_grad_%d" % i)


@requires_so2_cuda
def test_hsoft_gpu_fused_dispatch_forward_all_gradients(record_property, monkeypatch):
    from dptb.nn import so2_activation_routes as routes

    monkeypatch.setenv("DPTB_SO2_ACTIVATION_FUSED_P0", "1")
    torch.manual_seed(928)
    opts = dict(irreps_in="4x0e+3x1o+2x2e", irreps_out="3x0e+2x1o+2x2e",
                num_experts=4, num_shared_experts=1, mole_expert_parameterization="shared_core",
                mole_expert_rank=3, mole_linear_mode="cublas_grouped")
    staged = SO2_Linear(**opts, so2_fusion_mode="staged").cuda()
    fused = SO2_Linear(**opts, so2_fusion_mode="streamed_m_major_fused_p0").cuda()
    fused.load_state_dict(staged.state_dict(), strict=True)
    ra = MOLERouterV3(9, num_experts=4, top_k=4).cuda()
    rb = copy.deepcopy(ra)
    xa = torch.randn(31, staged.irreps_in.dim, device="cuda", requires_grad=True)
    xb = xa.detach().clone().requires_grad_(True)
    ha = torch.randn(31, 9, device="cuda", requires_grad=True)
    hb = ha.detach().clone().requires_grad_(True)
    r = torch.randn(31, 3, device="cuda")
    a = staged(xa, r, _route(ra, ha))[0]
    before = routes.STATS.calls.get(routes.FUSED_P0, 0)
    b = fused(xb, r, _route(rb, hb))[0]
    calls = routes.STATS.calls.get(routes.FUSED_P0, 0) - before
    assert calls > 0
    record_property("observed_fused_p0_calls", calls)
    _check(a, b, record_property, "gpu_fp32_forward", 3e-4, 3e-4)
    ga = torch.autograd.grad(a.square().sum(), [xa, ha, *ra.parameters(), *staged.parameters()])
    gb = torch.autograd.grad(b.square().sum(), [xb, hb, *rb.parameters(), *fused.parameters()])
    for i, (aa, bb) in enumerate(zip(ga, gb)):
        _check(aa, bb, record_property, "gpu_fp32_grad_%d" % i, 2e-3, 2e-3)
