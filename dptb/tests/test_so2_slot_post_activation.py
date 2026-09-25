"""Nonlinear experts for per-edge top-k routing (so2_expert_mixing_mode='post_activation_slot').

h' = sum_j g_j act(SO2_{e_j}(x)), each top-k slot through the layer's own activation-space route (shared expert
folded in, coefficient 1), the router coefficients g applied after the activation."""
import pytest
import torch
from e3nn import o3

from dptb.nn.embedding.lem_moe_v3_plugins import build_gate_activation
from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLELinear, SO2_Linear, SO2SlotPostActivationMixer
from dptb.tests._requires import requires_so2_cuda
from dptb.tests.model_helpers import _build, _data

IRREPS_IN = "4x0e + 3x1o + 2x2e"
GATED_OUT = "3x0e + 2x1o + 2x2e"


def _layer(route, irreps_out, *, radial=False, interpolation=False, shared=1, dtype=torch.float64, device="cpu"):
    torch.manual_seed(20260925)
    layer = SO2_Linear(irreps_in=IRREPS_IN, irreps_out=irreps_out, radial_emb=radial,
                       latent_dim=8 if radial else None, radial_channels=[16] if radial else None,
                       use_interpolation=interpolation, num_experts=4, num_shared_experts=shared,
                       mole_linear_mode="indexed_ref", so2_fusion_mode=route)
    return layer.to(device=device, dtype=dtype)


def _routing(n, num_experts, k=2, device="cpu", dtype=torch.float64, seed=7):
    """As MOLERouterV3 builds it: coefficients = softmax over the selected logits (leaf: g.selected_logits)."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n, num_experts, generator=g, dtype=torch.float64).to(device=device, dtype=dtype)
    sel, idx = logits.topk(k, dim=-1)
    sel = sel.detach().requires_grad_(True)
    val = sel.softmax(-1)
    coeffs = torch.zeros(n, num_experts, dtype=dtype, device=device).scatter(1, idx, val)
    mg = MOLEGlobals(coefficients=coeffs, sizes=None, topk_indices=idx, topk_values=val,
                     activation_space=True, coefficients_sum_to_one=True)
    mg.selected_logits = sel
    return mg


def _inputs(layer, n=23, device="cpu", dtype=torch.float64, seed=11):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, layer.irreps_in.dim, generator=g, dtype=torch.float64).to(device=device, dtype=dtype)
    R = torch.randn(n, 3, generator=g, dtype=torch.float64).to(device=device, dtype=dtype)
    lat = torch.randn(n, 8, generator=g, dtype=torch.float64).to(device=device, dtype=dtype)
    return x.requires_grad_(True), R, lat


CASES = {"plain": dict(), "radial": dict(radial=True), "interp": dict(interpolation=True),
         "shared0": dict(shared=0), "shared2": dict(shared=2)}


@pytest.mark.parametrize("route", ["staged", "streamed_m_major_cueq"])
@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_identity_activation_equals_pre_activation_mix(route, case):
    layer = _layer(route, GATED_OUT, **case)
    x, R, lat = _inputs(layer)
    g = _routing(x.shape[0], layer.num_experts)
    latents = lat if case.get("radial") else None
    ref, _ = layer(x, R, g, latents=latents)
    mixer = SO2SlotPostActivationMixer(layer, torch.nn.Identity())
    got, _ = mixer(x, R, g, latents=latents)
    torch.testing.assert_close(got, ref, rtol=1e-10, atol=1e-10)
    # router gradient w.r.t. the selected logits: non-MoLE blocks (interpolation) sit in every slot and add the same
    # term to d out / d g_j for all j, which the softmax Jacobian removes
    wanted = [x, g.selected_logits] + [p for p in layer.parameters()]
    ga = torch.autograd.grad(ref.square().sum(), wanted, allow_unused=True, retain_graph=True)
    gb = torch.autograd.grad(got.square().sum(), wanted, allow_unused=True)
    for a, b in zip(ga, gb):
        if a is None or b is None:
            assert a is None and b is None
            continue
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)


def _gate_layer(route="staged", dtype=torch.float64, device="cpu", **kw):
    gate = build_gate_activation(o3.Irreps(GATED_OUT))
    layer = _layer(route, str(gate.irreps_in), dtype=dtype, device=device, **kw)
    return layer, gate.to(device=device, dtype=dtype)


# Rotations only, and only where the SO2 layer itself is equivariant: the eSCN-type SO2 layer is SO(3)-equivariant
# but not inversion-equivariant (a plain pre-activation layer on natural-parity irreps fails x -> D(-I)x, R -> -R), and
# use_interpolation replaces the m>0 linears by an elementwise SiLU MLP on the (m, -m) pair in the edge frame, which
# does not commute with the roll about the edge axis (the base layer then fails the rotation check too).  The mixer
# adds an equivariant activation per slot, so it inherits exactly the layer's symmetry.
@pytest.mark.parametrize("case", [dict(), dict(radial=True), dict(shared=0), dict(shared=2)],
                         ids=["plain", "radial", "shared0", "shared2"])
def test_gate_mixer_is_rotation_equivariant(case):
    layer, gate = _gate_layer(**case)
    mixer = SO2SlotPostActivationMixer(layer, gate)
    x, R, lat = _inputs(layer)
    g = _routing(x.shape[0], layer.num_experts)       # routing is an invariant input
    latents = lat if case.get("radial") else None
    torch.manual_seed(3)
    Q = o3.rand_matrix(dtype=torch.float64)
    D_in = layer.irreps_in.D_from_matrix(Q)
    D_out = gate.irreps_out.D_from_matrix(Q)
    # the layer takes edge vectors in (x, y, z) while its l=1 features transform with e3nn's D(Q); the matching
    # rotation of the edge vectors is Qc = P^T Q P with P the (y, z, x) permutation (DeePTB's coordinate convention)
    P = torch.eye(3, dtype=torch.float64)[[1, 2, 0]]
    Qc = P.T @ Q @ P
    out, _ = mixer(x, R, g, latents=latents)
    out_rot, _ = mixer(x @ D_in.T, R @ Qc.T, g, latents=latents)
    # the SO2 layer itself reaches ~1e-6 in float64 (float32 inside its Wigner path), so compare at 1e-5
    torch.testing.assert_close(out_rot, out @ D_out.T, rtol=1e-5, atol=1e-5)
    ref_rot, _ = layer(x @ D_in.T, R @ Qc.T, g, latents=latents)                 # the pre-activation layer, same check
    torch.testing.assert_close(ref_rot, layer(x, R, g, latents=latents)[0] @ layer.irreps_out.D_from_matrix(Q).T,
                               rtol=1e-5, atol=1e-5)


def test_gate_mixer_is_not_the_pre_activation_mix():
    layer, gate = _gate_layer()
    x, R, _ = _inputs(layer)
    g = _routing(x.shape[0], layer.num_experts)
    post, _ = SO2SlotPostActivationMixer(layer, gate)(x, R, g)
    pre = gate(layer(x, R, g)[0])
    assert (post - pre).abs().max() > 1e-3               # activation inside the sum is a different function


def test_gradcheck_float64():
    layer, gate = _gate_layer(radial=True)
    mixer = SO2SlotPostActivationMixer(layer, gate)
    x, R, lat = _inputs(layer, n=5)
    g = _routing(5, layer.num_experts)
    idx = g.topk_indices

    def f(x_, val_, lat_):
        coeffs = torch.zeros(5, layer.num_experts, dtype=val_.dtype).scatter(1, idx, val_)
        mg = MOLEGlobals(coefficients=coeffs, sizes=None, topk_indices=idx, topk_values=val_,
                         activation_space=True, coefficients_sum_to_one=True)
        return mixer(x_, R, mg, latents=lat_)[0]

    assert torch.autograd.gradcheck(f, (x.detach().requires_grad_(True), g.topk_values.detach().clone().requires_grad_(True),
                                        lat.detach().requires_grad_(True)), eps=1e-6, atol=1e-6, rtol=1e-5)


def test_router_coefficients_receive_gradient_through_the_activated_slots():
    layer, gate = _gate_layer()
    x, R, _ = _inputs(layer)
    g = _routing(x.shape[0], layer.num_experts)
    out, _ = SO2SlotPostActivationMixer(layer, gate)(x, R, g)
    (grad,) = torch.autograd.grad(out.square().sum(), [g.selected_logits])
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_no_per_edge_weight_is_built(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("per-row weight materialised")

    monkeypatch.setattr(MOLELinear, "_mix_expert_parameters", refuse)
    monkeypatch.setattr(MOLELinear, "_apply_expert_indexed_ref", refuse)
    monkeypatch.setattr(MOLELinear, "apply_experts", refuse)
    for route in ("staged", "streamed_m_major_cueq"):
        layer, gate = _gate_layer(route)
        x, R, _ = _inputs(layer)
        g = _routing(x.shape[0], layer.num_experts)
        out, _ = SO2SlotPostActivationMixer(layer, gate)(x, R, g)
        out.square().sum().backward()


def test_rejects_routing_it_cannot_fold():
    layer, gate = _gate_layer()
    mixer = SO2SlotPostActivationMixer(layer, gate)
    x, R, _ = _inputs(layer)
    g = _routing(x.shape[0], layer.num_experts)
    g.coefficients_sum_to_one = False
    with pytest.raises(ValueError, match="sum to one"):
        mixer(x, R, g)
    with pytest.raises(ValueError, match="activation-space"):
        mixer(x, R, MOLEGlobals(coefficients=g.coefficients, sizes=None))


def _route_counters():
    """(fused slot calls, pack/scatter calls) of whichever SO2CUDA activation module this library has."""
    try:
        from dptb.nn import so2_activation_routes as r
        return r.STATS.calls[r.FUSED_P0], r.STATS.calls[r.PACK_SCATTER]
    except ImportError:                                   # release 0923 (07c0711): separate modules
        from dptb.nn import so2_activation_fused_p0 as f0
        from dptb.nn import top1_so2_cuda as t1
        pack = getattr(t1, "ACTIVATION_CALLS", getattr(t1, "CALLS", 0))
        return f0.CALLS, pack


@requires_so2_cuda
@pytest.mark.parametrize("force_pack_scatter", [False, True], ids=["fused_p0", "pack_scatter"])
@pytest.mark.parametrize("case", [dict(), dict(radial=True), dict(radial=True, back=True), dict(interpolation=True),
                                  dict(shared=0)], ids=["plain", "radial", "radial-back", "interp", "shared0"])
def test_fused_route_matches_staged_route_on_cuda(monkeypatch, case, force_pack_scatter):
    case = dict(case)
    back = case.pop("back", False)
    if force_pack_scatter:
        monkeypatch.setenv("DPTB_SO2_ACTIVATION_FUSED_P0", "0")
    if back:                                              # radial after the linear: fewer output than input channels
        gate = build_gate_activation(o3.Irreps("2x0e + 1x1o + 1x2e"))
        ref_layer = _layer("staged", str(gate.irreps_in), dtype=torch.float32, device="cuda", **case)
        got_layer = _layer("streamed_m_major_fused_p0", str(gate.irreps_in), dtype=torch.float32, device="cuda", **case)
        gate = gate.to(device="cuda", dtype=torch.float32)
    else:
        ref_layer, gate = _gate_layer("staged", dtype=torch.float32, device="cuda", **case)
        got_layer, _ = _gate_layer("streamed_m_major_fused_p0", dtype=torch.float32, device="cuda", **case)
    got_layer.load_state_dict(ref_layer.state_dict())
    x, R, lat = _inputs(ref_layer, n=301, device="cuda", dtype=torch.float32)
    g = _routing(x.shape[0], ref_layer.num_experts, device="cuda", dtype=torch.float32)
    latents = lat.requires_grad_(True) if case.get("radial") else None
    outs, grads = [], []
    for layer in (ref_layer, got_layer):
        before = _route_counters()
        out, _ = SO2SlotPostActivationMixer(layer, gate)(x, R, g, latents=latents)
        after = _route_counters()
        if layer is got_layer:                            # the CUDA route really ran, once per top-k slot
            if force_pack_scatter:
                assert after[1] - before[1] == g.topk_indices.shape[1], (before, after)
            else:
                assert after[0] - before[0] == g.topk_indices.shape[1], (before, after)
        params = list(layer.parameters())
        wanted = [x, g.selected_logits] + ([latents] if latents is not None else []) + params
        grads.append(torch.autograd.grad(out.square().sum(), wanted, retain_graph=True))
        outs.append(out.detach())
    torch.testing.assert_close(outs[1], outs[0], rtol=1e-4, atol=1e-5)
    for a, b in zip(grads[1], grads[0]):
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-4)


PA = dict(edge_router_prior_activate=True, num_experts=4, num_shared_experts=1, top_k=2, so2_fusion_mode="staged")


def test_prior_activate_embedding_builds_and_trains_with_post_activation_slot():
    model = _build(False, **dict(PA, so2_expert_mixing_mode="post_activation_slot"))
    mixers = [m.post_activation_expert_mixer for m in model.embedding.modules()
              if getattr(m, "post_activation_expert_mixer", None) is not None]
    assert mixers and all(isinstance(m, SO2SlotPostActivationMixer) for m in mixers)
    out = model(_data(model))
    (out["node_features"].square().mean() + out["edge_features"].square().mean()).backward()
    assert model.embedding.router.net[0].weight.grad is not None
    assert model.embedding.router.net[0].weight.grad.abs().sum() > 0


def test_post_activation_slot_state_dict_matches_pre_activation():
    pre = _build(False, **PA)
    post = _build(False, **dict(PA, so2_expert_mixing_mode="post_activation_slot"))
    assert pre.state_dict().keys() == post.state_dict().keys()
    post.load_state_dict(pre.state_dict(), strict=True)


def test_prior_activate_still_rejects_the_graph_level_post_activation_mixer():
    with pytest.raises(ValueError, match="so2_expert_mixing_mode"):
        _build(False, **dict(PA, so2_expert_mixing_mode="post_activation"))


def test_empty_rows_return_the_activated_width():
    layer, gate = _gate_layer("streamed_m_major_cueq")
    x = torch.zeros(0, layer.irreps_in.dim, dtype=torch.float64)
    R = torch.zeros(0, 3, dtype=torch.float64)
    empty = MOLEGlobals(coefficients=torch.zeros(0, layer.num_experts, dtype=torch.float64), sizes=None)
    out, _ = SO2SlotPostActivationMixer(layer, gate)(x, R, empty)       # globals as the router builds them for 0 edges
    assert out.shape == (0, gate.irreps_out.dim)
    ident, _ = SO2SlotPostActivationMixer(layer, torch.nn.Identity())(x, R, empty)
    assert ident.shape == (0, layer.irreps_out.dim)


@pytest.mark.parametrize("case", [dict(), dict(radial=True), dict(interpolation=True), dict(shared=2)],
                         ids=["plain", "radial", "interp", "shared2"])
def test_matches_an_independent_weight_space_oracle(case):
    """Per slot, the same layer on its weight-space route with one-hot per-row coefficients (one [out, in] weight per row,
    shared expert added by _mix_expert_parameters), then the Gate, then the router coefficients."""
    layer, gate = _gate_layer(**case)
    x, R, lat = _inputs(layer)
    g = _routing(x.shape[0], layer.num_experts)
    latents = lat if case.get("radial") else None
    got, _ = SO2SlotPostActivationMixer(layer, gate)(x, R, g, latents=latents)
    n = x.shape[0]
    ref = 0
    for j in range(g.topk_indices.shape[1]):
        onehot = torch.zeros(n, layer.num_experts, dtype=x.dtype).scatter(1, g.topk_indices[:, j:j + 1], 1.0)
        ws = MOLEGlobals(coefficients=onehot, sizes=None, graph_index=torch.arange(n))
        y, _ = layer(x, R, ws, latents=latents)
        ref = ref + gate(y) * g.topk_values[:, j:j + 1]
    torch.testing.assert_close(got, ref, rtol=1e-10, atol=1e-10)
    wanted = [x, g.selected_logits] + list(layer.parameters())
    ga = torch.autograd.grad(ref.square().sum(), wanted, allow_unused=True, retain_graph=True)
    gb = torch.autograd.grad(got.square().sum(), wanted, allow_unused=True)
    for a, b in zip(ga, gb):
        if a is None or b is None:
            assert a is None and b is None
            continue
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)


def test_rejects_configurations_it_does_not_support():
    with pytest.raises(ValueError, match="per-edge routing"):          # graph-level / non-prior_activate routing
        _build(False, **dict(PA, edge_router_prior_activate=False, so2_expert_mixing_mode="post_activation_slot"))
    with pytest.raises(ValueError, match="so2_expert_route_checkpoint"):
        _build(False, **dict(PA, so2_expert_mixing_mode="post_activation_slot", so2_expert_route_checkpoint=True))
    with pytest.raises(ValueError, match="Switch"):
        _build(False, **dict(PA, num_experts=4, top_k=1, num_shared_experts=0, edge_router_top1_mode="switch",
                             so2_fusion_mode="streamed_m_major_cueq", so2_expert_mixing_mode="post_activation_slot"))
