"""Shared expert bases: routed numerics, gradients, checkpoint and optimizer semantics."""
import copy
import io

import pytest
import torch
from torch.nn import functional as F

from dptb.nn.tensor_product_moe_v3 import MOLEGlobals, MOLELinear, SO2_Linear
from dptb.tests._requires import requires_cuda, requires_so2_cuda
from dptb.tests.model_helpers import _build, _data
from dptb.utils.dpa4_optim import HybridMuon


def _linear(**kw):
    return MOLELinear(7, 5, num_experts=4, num_shared_experts=1,
                      mole_expert_parameterization="shared_core", mole_expert_rank=3, **kw)


def _globals(idx, val, normalized=True):
    coeff = val.new_zeros(idx.shape[0], 4).scatter(1, idx, val)
    return MOLEGlobals(coefficients=coeff, topk_indices=idx, topk_values=val,
                       activation_space=True, coefficients_sum_to_one=normalized)


@pytest.mark.parametrize("pair_rows", [False, True])
@pytest.mark.parametrize("normalized", [False, True])
def test_shared_core_matches_factorized_output_and_all_gradients(pair_rows, normalized):
    torch.manual_seed(31)
    lin = _linear(bias=not pair_rows, mole_linear_mode="split_loop").double()
    x = torch.randn((6, 2, 7) if pair_rows else (6, 7), dtype=torch.float64, requires_grad=True)
    # Expert 3 is empty; m>0-style inputs have two matrix rows per route.
    idx = torch.tensor([[0, 1], [2, 0], [1, 2], [2, 1], [0, 2], [1, 0]])
    route_leaf = torch.rand(6, 2, dtype=torch.float64, requires_grad=True)
    # Folding shared weights is equivalent only on the probability simplex.
    # Differentiate normalized routing through its logits, as the real router
    # does; independent coefficient derivatives include off-simplex directions.
    val = route_leaf.softmax(-1) if normalized else route_leaf
    got = lin(x, _globals(idx, val, normalized))
    z = x @ lin.basis_right
    ref = F.linear(x, lin.weight_shared.sum(0), lin.bias_shared.sum(0) if lin.bias_shared is not None else None)
    for j in range(2):
        out = torch.einsum("b...r,bsr->b...s", z, lin.core_experts[idx[:, j]]) @ lin.basis_left.t()
        if lin.bias_experts is not None:
            out = out + lin.bias_experts[idx[:, j]]
        ref = ref + out * val[:, j].reshape(6, *([1] * (x.dim() - 1)))
    torch.testing.assert_close(got, ref, atol=1e-11, rtol=1e-11)
    args = [x, route_leaf, *lin.parameters()]
    ga = torch.autograd.grad(got.square().sum(), args, retain_graph=normalized)
    gb = torch.autograd.grad(ref.square().sum(), args)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
    core_index = next(i for i, p in enumerate(args) if p is lin.core_experts)
    assert torch.count_nonzero(ga[core_index][3]) == 0


def test_full_default_keeps_legacy_state_and_initialization():
    torch.manual_seed(42)
    legacy = MOLELinear(7, 5, num_experts=4)
    torch.manual_seed(42)
    explicit = MOLELinear(7, 5, num_experts=4, mole_expert_parameterization="full")
    assert set(legacy.state_dict()) == {"weight_experts", "bias_experts", "weight_shared", "bias_shared"}
    for key, value in legacy.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)


def test_rank_validation_and_shared_core_checkpoint_roundtrip():
    for bad in (0, -1, True, 2.5):
        with pytest.raises(ValueError):
            MOLELinear(3, 2, mole_expert_parameterization="shared_core", mole_expert_rank=bad)
    lin = MOLELinear(3, 2, mole_expert_parameterization="shared_core", mole_expert_rank=64)
    assert lin.core_experts.shape[-2:] == (2, 2)
    assert "weight_experts" not in lin.state_dict()
    buffer = io.BytesIO()
    torch.save(lin.state_dict(), buffer)
    buffer.seek(0)
    other = MOLELinear(3, 2, mole_expert_parameterization="shared_core", mole_expert_rank=64)
    other.load_state_dict(torch.load(buffer, weights_only=True))
    x = torch.randn(6, 3)
    torch.testing.assert_close(lin(x, None), other(x, None), rtol=0, atol=0)
    before = lin.weight_experts.detach().clone()
    lin.scale_expert_weights_(0.5)
    torch.testing.assert_close(lin.weight_experts, before * 0.5)
    with pytest.raises(RuntimeError):
        MOLELinear(3, 2).load_state_dict(lin.state_dict())


def test_zero_core_keeps_shared_function_and_can_start_learning():
    lin = _linear().double()
    with torch.no_grad():
        lin.core_experts.zero_()
        lin.bias_experts.zero_()
    x = torch.randn(4, 7, dtype=torch.float64)
    idx = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
    val = torch.full((4, 2), 0.5, dtype=torch.float64)
    out = lin(x, _globals(idx, val))
    torch.testing.assert_close(out, F.linear(x, lin.weight_shared.sum(0), lin.bias_shared.sum(0)))
    out.square().sum().backward()
    assert lin.core_experts.grad.norm() > 0
    assert torch.count_nonzero(lin.basis_left.grad) == 0
    assert torch.count_nonzero(lin.basis_right.grad) == 0


def test_expert_update_scaling_applies_to_core_and_not_shared_bases():
    a = _linear(bias=False)
    b = copy.deepcopy(a)
    oa = HybridMuon(a.named_parameters(), lr=0.01, weight_decay=0, muon_clip=False)
    ob = HybridMuon(b.named_parameters(), lr=0.01, weight_decay=0, muon_clip=False,
                    expert_update_scale="const", expert_update_scale_const=0.25)
    before = {name: p.detach().clone() for name, p in a.named_parameters()}
    for pa, pb in zip(a.parameters(), b.parameters()):
        grad = torch.randn_like(pa)
        pa.grad, pb.grad = grad, grad.clone()
    oa.step()
    ob.step()
    for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        factor = 0.25 if name == "core_experts" else 1.0
        torch.testing.assert_close(before[name] - pb, factor * (before[name] - pa), rtol=2e-4, atol=2e-7)


def test_model_builder_propagates_shared_core_and_reload():
    cfg = dict(num_experts=4, num_shared_experts=1, top_k=2, edge_router_prior_activate=True,
               so2_fusion_mode="staged", mole_expert_parameterization="shared_core", mole_expert_rank=3)
    model = _build(False, **cfg)
    linears = [m for m in model.modules() if isinstance(m, MOLELinear)]
    assert linears and all(m.mole_expert_parameterization == "shared_core" for m in linears)
    model.eval()
    out = model(_data(model))
    loss = out["node_features"].square().mean() + out["edge_features"].square().mean()
    loss.backward()
    assert any(m.core_experts.grad is not None and m.core_experts.grad.norm() > 0 for m in linears)
    other = _build(False, **cfg).eval()
    other.load_state_dict(model.state_dict())
    out2 = other(_data(other))
    for key in ("node_features", "edge_features"):
        torch.testing.assert_close(out[key], out2[key])


def test_complete_model_preserves_all_common_initial_parameters():
    cfg = dict(num_experts=4, num_shared_experts=1, top_k=2, edge_router_prior_activate=True,
               so2_fusion_mode="staged", tp_radial_emb=True, tp_radial_channels=[8])
    torch.manual_seed(20260926)
    full = _build(False, **cfg)
    torch.manual_seed(20260926)
    core = _build(False, **cfg, mole_expert_parameterization="shared_core", mole_expert_rank=3)
    full_params, core_params = dict(full.named_parameters()), dict(core.named_parameters())
    common = set(full_params) & set(core_params)
    assert any("router" in name for name in common)
    assert any("radial_emb" in name for name in common)
    assert any(name.endswith("weight_shared") for name in common)
    # All legacy parameters except the replaced routed matrices must remain,
    # including routed biases and the entire graph/shared/output backbone.
    assert set(full_params) - common == {name for name in full_params if name.endswith("weight_experts")}
    for name in common:
        assert torch.equal(full_params[name], core_params[name]), name


def test_concat_prior_rebuild_respects_selective_layer_placement():
    model = _build(False, num_experts=4, num_shared_experts=1, top_k=2,
                   edge_router_prior_activate=True, so2_fusion_mode="staged",
                   so2_moe_layers=[1], so2_expert_mixing_mode="post_activation_shared")
    first = [m for m in model.embedding.layers[0].modules() if isinstance(m, MOLELinear)]
    assert first and all(m.num_experts == 0 for m in first)
    assert all(m.weight_shared is not None for m in first)
    assert not any("experts" in n for n, _ in model.embedding.layers[0].named_parameters())
    out = model(_data(model))
    loss = out["node_features"].square().mean() + out["edge_features"].square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert any(m.weight_shared.grad is not None and m.weight_shared.grad.norm() > 0 for m in first)


@requires_cuda
def test_default_cuda_device_preserves_shared_initialization_and_rng():
    with torch.device("cuda"):
        torch.manual_seed(20260926)
        full = MOLELinear(7, 5, num_experts=4, num_shared_experts=1)
        full_next = torch.rand(11, device="cuda")
        full_cpu_next = torch.rand(11, device="cpu")
        torch.manual_seed(20260926)
        core = _linear()
        core_next = torch.rand(11, device="cuda")
        core_cpu_next = torch.rand(11, device="cpu")
    assert core.core_experts.is_cuda
    for name in ("weight_shared", "bias_shared", "bias_experts"):
        assert torch.equal(getattr(full, name), getattr(core, name)), name
    assert torch.equal(full_next, core_next)
    assert torch.equal(full_cpu_next, core_cpu_next)


@requires_so2_cuda
def test_shared_core_fused_p0_matches_staged_gradients():
    from dptb.nn import so2_activation_routes as routes
    opts = dict(irreps_in="4x0e+3x1o+2x2e", irreps_out="3x0e+2x1o+2x2e",
                num_experts=4, num_shared_experts=1, mole_expert_parameterization="shared_core",
                mole_expert_rank=2, mole_linear_mode="cublas_grouped")
    ref = SO2_Linear(**opts, so2_fusion_mode="staged").cuda()
    fused = SO2_Linear(**opts, so2_fusion_mode="streamed_m_major_fused_p0").cuda()
    fused.load_state_dict(ref.state_dict())
    x = torch.randn(17, ref.irreps_in.dim, device="cuda", requires_grad=True)
    r = torch.randn(17, 3, device="cuda")
    logits = torch.randn(17, 4, device="cuda")
    val, idx = logits.topk(2, -1)
    val = val.softmax(-1).detach().requires_grad_(True)
    before = routes.STATS.calls[routes.FUSED_P0]
    ya = ref(x, r, _globals(idx, val))[0]
    yb = fused(x, r, _globals(idx, val))[0]
    assert routes.STATS.calls[routes.FUSED_P0] > before
    torch.testing.assert_close(ya, yb, atol=3e-4, rtol=3e-4)
    ga = torch.autograd.grad(ya.square().sum(), [x, val, *ref.parameters()])
    gb = torch.autograd.grad(yb.square().sum(), [x, val, *fused.parameters()])
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, atol=2e-3, rtol=2e-3)
