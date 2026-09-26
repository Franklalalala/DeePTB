"""DPA3-MoE style options (route 0925): edge_router_gate='full_softmax' and so2_expert_mixing_mode='post_activation_shared'.

Liu et al., "Mixture of experts architectures for machine learning interatomic potentials", npj Artif. Intell. (2026),
eqs. 4-6: y = sum_j a_j act(W_j x + b_j) + act(W_sh x + b_sh), a = top-k entries of softmax over the routed experts
(not renormalised).  Defaults stay the production router / pre_activation mixing.
"""
from __future__ import annotations

import pytest
import torch
from e3nn import o3
from torch.nn import functional as F

from dptb.nn.embedding.lem_moe_v3_plugins import build_gate_activation
from dptb.nn.tensor_product_moe_v3 import (MOLEGlobals, MOLELinear, MOLERouterV3, SO2_Linear,
                                           SO2SharedPostActivationMixer)


def _close(a, b, tol=1e-5):
    torch.testing.assert_close(a, b, rtol=tol, atol=tol)


def _router(**kw):
    torch.manual_seed(0)
    r = MOLERouterV3(8, num_experts=6, top_k=2, logit_kind="cosine", select="logit", **kw)
    return r.eval()


def _globals(idx, val, num_experts, branch="all", sum_to_one=False):
    coeff = torch.zeros(idx.shape[0], num_experts, dtype=val.dtype).scatter(1, idx, val)
    return MOLEGlobals(coefficients=coeff, topk_indices=idx, topk_values=val, activation_space=True,
                       coefficients_sum_to_one=sum_to_one, branch=branch)


# ---------------------------------------------------------------- router gate

def test_full_softmax_gate_keeps_the_probability_mass():
    r = _router(gate="full_softmax")
    x = torch.randn(11, 8)
    coeffs, _, _ = r(x)
    idx, val = r.last_topk()
    p = torch.softmax(r._logits(x), dim=-1)
    _close(val, p.gather(1, idx))
    _close(coeffs.gather(1, idx), val)
    assert (coeffs.sum(-1) < 1.0).all()
    assert not r.coefficients_sum_to_one and r.router_config()["gate"] == "full_softmax"


def test_renorm_gate_is_the_default_and_unchanged():
    a, b = _router(), _router(gate="renorm")
    x = torch.randn(9, 8)
    ca, _, _ = a(x)
    cb, _, _ = b(x)
    _close(ca, cb, 0.0)
    idx, val = a.last_topk()
    _close(val, torch.softmax(a._logits(x).gather(1, idx), dim=-1))
    _close(ca.sum(-1), torch.ones(9))
    assert a.coefficients_sum_to_one and a.router_config()["gate"] == "renorm"


def test_router_rejects_unknown_gate():
    with pytest.raises(ValueError, match="gate"):
        MOLERouterV3(8, num_experts=6, top_k=2, gate="sparsemax")


# ---------------------------------------------------------------- MOLELinear branches

def test_mole_linear_branches_add_up():
    torch.manual_seed(1)
    lin = MOLELinear(5, 4, num_experts=3, num_shared_experts=1, bias=True, mole_linear_mode="split_loop")
    x = torch.randn(7, 5)
    idx = torch.stack([torch.randperm(3)[:2] for _ in range(7)])
    val = torch.rand(7, 2)

    def routed(v):
        return sum(v[:, j:j + 1] * (torch.einsum("noi,ni->no", lin.weight_experts[idx[:, j]], x)
                                     + lin.bias_experts[idx[:, j]]) for j in range(2))

    shared = F.linear(x, lin.weight_shared.sum(0), lin.bias_shared.sum(0))
    _close(lin(x, _globals(idx, val, 3, "shared")), shared)
    _close(lin(x, _globals(idx, val, 3, "routed")), routed(val))
    _close(lin(x, _globals(idx, val, 3, "all")), routed(val) + shared)
    # coefficients summing to one fold the shared expert into the slots, but only in an "all" pass
    vn = val / val.sum(-1, keepdim=True)
    _close(lin(x, _globals(idx, vn, 3, "all", True)), routed(vn) + shared)
    _close(lin(x, _globals(idx, vn, 3, "routed", True)), routed(vn))


def test_branch_needs_activation_space():
    with pytest.raises(ValueError, match="activation-space"):
        MOLEGlobals(coefficients=torch.zeros(2, 3), branch="routed")
    with pytest.raises(ValueError, match="branch"):
        MOLEGlobals(coefficients=torch.zeros(2, 3), activation_space=True, branch="experts")


# ---------------------------------------------------------------- SO2 layer + shared post-activation mixer

def _so2(use_interp, seed=2):
    torch.manual_seed(seed)
    act = build_gate_activation(o3.Irreps("3x0e+2x1o+2x2e"))
    tp = SO2_Linear(irreps_in=o3.Irreps("4x0e+3x1o+2x2e"), irreps_out=act.irreps_in, num_experts=4,
                    num_shared_experts=1, use_interpolation=use_interp, so2_fusion_mode="staged")
    return tp, act


def _inputs(tp, n=9, seed=3):
    torch.manual_seed(seed)
    x = torch.randn(n, tp.irreps_in.dim)
    R = torch.randn(n, 3)
    idx = torch.stack([torch.randperm(4)[:2] for _ in range(n)])
    val = torch.rand(n, 2) * 0.5
    return x, R, idx, val


@pytest.mark.parametrize("use_interp", [False, True])
def test_shared_mixer_with_identity_activation_is_one_pass(use_interp):
    """act = identity: shared pass + routed passes == the single all-pass (non-MoLE blocks counted once)."""
    tp, _ = _so2(use_interp)
    x, R, idx, val = _inputs(tp)
    g = _globals(idx, val, 4)
    out, _ = SO2SharedPostActivationMixer(tp, torch.nn.Identity())(x, R, g)
    ref, _ = tp(x, R, g)
    _close(out, ref)


@pytest.mark.parametrize("use_interp", [False, True])
def test_shared_mixer_with_zero_routed_experts_is_the_dense_layer(use_interp):
    tp, act = _so2(use_interp)
    with torch.no_grad():
        for m in tp.modules():
            if isinstance(m, MOLELinear):
                m.weight_experts.zero_()
                if m.bias_experts is not None:
                    m.bias_experts.zero_()
    x, R, idx, val = _inputs(tp)
    out, _ = SO2SharedPostActivationMixer(tp, act)(x, R, _globals(idx, val, 4))
    dense, _ = tp(x, R, _globals(idx, val, 4))
    _close(out, act(dense))


def test_shared_mixer_is_the_formula_and_trains():
    tp, act = _so2(True)
    x, R, idx, val = _inputs(tp)
    val = val.clone().requires_grad_(True)
    g = _globals(idx, val, 4)
    out, _ = SO2SharedPostActivationMixer(tp, act)(x, R, g)
    # explicit formula with the branch passes
    n = x.shape[0]
    sh, _ = tp(x, R, MOLEGlobals(coefficients=torch.zeros(n, 4), topk_indices=idx[:, :1], topk_values=torch.zeros(n, 1),
                                  activation_space=True, branch="shared"))
    ref = act(sh)
    for j in range(2):
        one = torch.ones(n, 1)
        yj, _ = tp(x, R, _globals(idx[:, j:j + 1], one, 4, "routed"))
        ref = ref + act(yj) * val[:, j:j + 1]
    _close(out, ref)
    out.square().sum().backward()
    assert val.grad is not None and torch.isfinite(val.grad).all() and val.grad.abs().sum() > 0
    lins = [m for m in tp.modules() if isinstance(m, MOLELinear)]
    assert all(m.weight_shared.grad is not None and m.weight_shared.grad.abs().sum() > 0 for m in lins)
    used = torch.unique(idx)
    assert all(m.weight_experts.grad[used].abs().sum() > 0 for m in lins)


# ---------------------------------------------------------------- embedding level

from dptb.tests.test_lem_moe_v3_prior_2b import _build, _data  # noqa: E402

PA = dict(edge_router_prior_activate=True, num_experts=4, num_shared_experts=1, top_k=2, so2_fusion_mode="staged",
          edge_router_logit="cosine", edge_router_select="logit", edge_router_bias_at_eval=True)


def test_post_activation_shared_full_softmax_embedding_trains():
    model = _build(False, **dict(PA, so2_expert_mixing_mode="post_activation_shared", edge_router_gate="full_softmax"))
    emb = model.embedding
    assert emb.router.router_config()["gate"] == "full_softmax" and not emb.router.coefficients_sum_to_one
    out = model(_data(model))
    (out["node_features"].square().mean() + out["edge_features"].square().mean()).backward()
    assert emb.router.net[0].weight.grad is not None and emb.router.net[0].weight.grad.abs().sum() > 0
    lins = [m for m in emb.modules() if isinstance(m, MOLELinear)]
    assert lins and any(m.weight_experts.grad is not None and m.weight_experts.grad.abs().sum() > 0 for m in lins)


def test_zero_routed_experts_give_the_same_model_in_both_mixing_modes():
    """With the routed experts at zero, pre_activation (dense through the shared expert) and post_activation_shared
    compute the same function; the state dicts are interchangeable (upcycling starts from the dense model)."""
    pre = _build(False, **PA)
    post = _build(False, **dict(PA, so2_expert_mixing_mode="post_activation_shared", edge_router_gate="full_softmax"))
    post.load_state_dict(pre.state_dict())
    for model in (pre, post):
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, MOLELinear):
                    m.weight_experts.zero_()
                    if m.bias_experts is not None:
                        m.bias_experts.zero_()
        model.eval()
    a, b = pre(_data(pre)), post(_data(post))
    _close(a["edge_features"], b["edge_features"])
    _close(a["node_features"], b["node_features"])


def test_slot_mixer_rejects_the_full_softmax_gate():
    with pytest.raises(ValueError, match="post_activation_shared"):
        _build(False, **dict(PA, so2_expert_mixing_mode="post_activation_slot", edge_router_gate="full_softmax"))


# ---------------------------------------------------------------- the same mixer on the other SO2 routes

from dptb.tests._requires import requires_so2_cuda  # noqa: E402
from dptb.tests.test_so2_slot_post_activation import _gate_layer, _layer, _route_counters  # noqa: E402
from dptb.tests.test_so2_slot_post_activation import _inputs as _slot_inputs  # noqa: E402


def _full_softmax_routing(n, num_experts, k=2, device="cpu", dtype=torch.float64, seed=7):
    """As MOLERouterV3(gate='full_softmax') builds it: top-k entries of the softmax over every expert (sum < 1)."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, num_experts, generator=g, dtype=torch.float64).to(device=device, dtype=dtype)
    logits = logits.detach().requires_grad_(True)
    idx = logits.detach().topk(k, dim=-1).indices
    val = logits.softmax(-1).gather(1, idx)
    coeffs = torch.zeros(n, num_experts, dtype=dtype, device=device).scatter(1, idx, val)
    mg = MOLEGlobals(coefficients=coeffs, sizes=None, topk_indices=idx, topk_values=val,
                     activation_space=True, coefficients_sum_to_one=False)
    mg.logits = logits
    return mg


@pytest.mark.parametrize("route", ["staged", "streamed_m_major_cueq"])
@pytest.mark.parametrize("case", [dict(), dict(radial=True), dict(interpolation=True), dict(shared=0), dict(shared=2)],
                         ids=["plain", "radial", "interp", "shared0", "shared2"])
def test_identity_shared_mixer_equals_one_pass_on_each_route(route, case):
    layer = _layer(route, "3x0e + 2x1o + 2x2e", **case)
    x, R, lat = _slot_inputs(layer)
    g = _full_softmax_routing(x.shape[0], layer.num_experts)
    latents = lat if case.get("radial") else None
    ref, _ = layer(x, R, g, latents=latents)
    got, _ = SO2SharedPostActivationMixer(layer, torch.nn.Identity())(x, R, g, latents=latents)
    torch.testing.assert_close(got, ref, rtol=1e-10, atol=1e-10)


@requires_so2_cuda
@pytest.mark.parametrize("force_pack_scatter", [False, True], ids=["fused_p0", "pack_scatter"])
@pytest.mark.parametrize("case", [dict(), dict(radial=True), dict(interpolation=True), dict(shared=0), dict(shared=2)],
                         ids=["plain", "radial", "interp", "shared0", "shared2"])
def test_shared_mixer_fused_route_matches_staged_route_on_cuda(monkeypatch, case, force_pack_scatter):
    if force_pack_scatter:
        monkeypatch.setenv("DPTB_SO2_ACTIVATION_FUSED_P0", "0")
    ref_layer, gate = _gate_layer("staged", dtype=torch.float32, device="cuda", **case)
    got_layer, _ = _gate_layer("streamed_m_major_fused_p0", dtype=torch.float32, device="cuda", **case)
    got_layer.load_state_dict(ref_layer.state_dict())
    x, R, lat = _slot_inputs(ref_layer, n=301, device="cuda", dtype=torch.float32)
    g = _full_softmax_routing(x.shape[0], ref_layer.num_experts, device="cuda", dtype=torch.float32)
    latents = lat.requires_grad_(True) if case.get("radial") else None
    outs, grads = [], []
    for layer in (ref_layer, got_layer):
        before = _route_counters()
        out, _ = SO2SharedPostActivationMixer(layer, gate)(x, R, g, latents=latents)
        after = _route_counters()
        if layer is got_layer:          # the CUDA route really ran: the shared pass and one pass per top-k slot
            k1 = g.topk_indices.shape[1] + 1
            if force_pack_scatter:
                assert after[1] - before[1] == k1, (before, after)
            else:
                assert after[0] - before[0] == k1, (before, after)
        params = list(layer.parameters())
        wanted = [x, g.logits] + ([latents] if latents is not None else []) + params
        grads.append(torch.autograd.grad(out.square().sum(), wanted, retain_graph=True, allow_unused=True))
        outs.append(out.detach())
    torch.testing.assert_close(outs[1], outs[0], rtol=1e-4, atol=1e-5)
    for a, b in zip(grads[1], grads[0]):
        if a is None or b is None:
            assert a is None and b is None
            continue
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-4)
